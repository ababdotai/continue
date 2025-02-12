// @ts-nocheck
import { AIMessage, BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import dotenv from "dotenv";
import { ChatOllama } from "@langchain/ollama";
import { ChatDeepSeek } from "@langchain/deepseek";
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph } from "@langchain/langgraph";
import { MemorySaver, Annotation, messagesStateReducer } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import * as fs from "fs";
import * as path from "path";

// Load environment variables from .env file
dotenv.config();

// Get configuration from environment variables
const MODEL_PROVIDER = process.env.MODEL_PROVIDER || "ollama"; // "ollama" or "deepseek" or "openai"
const MODEL_NAME = process.env.MODEL_NAME || "qwen2.5"; // model must support tools
const TEMPERATURE = parseFloat(process.env.TEMPERATURE || "0.7");
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || "http://localhost:11434";
const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL;
const WORKSPACE_ROOT = process.env.WORKSPACE_ROOT || process.cwd();
console.log("Workspace root:", WORKSPACE_ROOT);
console.log("Using model provider:", MODEL_PROVIDER);
console.log("Using model:", MODEL_NAME);

// Define the graph state
const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
  }),
});

// Define the tools for the agent to use
const searchCodeTool = tool(async (args: { query: string }) => {
  // This is a placeholder for semantic code search implementation
  return "Found relevant code in file.ts: function example() {...}";
}, {
  name: "search_code",
  description: "Search for relevant code in the repository using semantic search.",
  schema: z.object({
    query: z.string().describe("The search query to find relevant code."),
  }),
});

const readFileTool = tool(async (args: { filepath: string }) => {
  try {
    const fullPath = path.join(WORKSPACE_ROOT, args.filepath);
    const content = fs.readFileSync(fullPath, 'utf-8');
    return content;
  } catch (error: any) {
    return `Error reading file: ${error.message}`;
  }
}, {
  name: "read_file",
  description: "Read the contents of a file in the repository.",
  schema: z.object({
    filepath: z.string().describe("The relative path to the file from workspace root."),
  }),
});

const editFileTool = tool(async (args: { filepath: string, content: string }) => {
  try {
    const fullPath = path.join(WORKSPACE_ROOT, args.filepath);
    fs.writeFileSync(fullPath, args.content, 'utf-8');
    return `Successfully edited file: ${args.filepath}`;
  } catch (error: any) {
    return `Error editing file: ${error.message}`;
  }
}, {
  name: "edit_file",
  description: "Edit or create a file in the repository.",
  schema: z.object({
    filepath: z.string().describe("The relative path to the file from workspace root."),
    content: z.string().describe("The new content to write to the file."),
  }),
});

const tools = [searchCodeTool, readFileTool, editFileTool];
const toolNode = new ToolNode(tools);

const systemPrompt = `You are a knowledgeable and helpful repository agent, designed to assist users in understanding and working with codebases.

ROLE:
- You are an expert in code analysis, repository navigation, and code modification
- You aim to provide accurate, detailed, and actionable responses
- You maintain a professional and helpful demeanor
- You MUST use the available tools to gather information before responding

AVAILABLE TOOLS:
1. search_code: Search for relevant code snippets using semantic search
   - Use this tool to find relevant code sections when answering questions
   - ALWAYS use this as your first step to understand the codebase

2. read_file: Read the contents of specific files
   - Use this after finding relevant files through search_code
   - Read files to understand implementation details

3. edit_file: Modify or create files in the repository
   - Use this only when explicitly asked to make changes
   - Always verify changes before applying

WORKFLOW GUIDELINES:
1. ALWAYS start by using search_code to find relevant information
2. Use read_file to examine files found through search
3. Never make assumptions without checking the code first
4. Provide explanations based on actual code, not assumptions
5. When asked about the project:
   - Search for README files
   - Look for package.json or similar config files
   - Search for main entry points
   - Examine project structure

PROJECT INFORMATION:
- Workspace Root: ${WORKSPACE_ROOT}
- You have full access to navigate and analyze the codebase

RESPONSE LANGUAGE:
- Match your response language to the user's query language
- Use English for code, comments, and technical terms
- Maintain consistent formatting and clear structure

IMPORTANT: You MUST use the tools to gather information before responding. Do not make assumptions or ask questions without first attempting to find the answers using the available tools.`;

// Define model configurations
const modelConfigs = {
  ollama: {
    model: MODEL_NAME,
    temperature: TEMPERATURE,
    baseUrl: OLLAMA_BASE_URL,
    numCtx: 8192,
    keepAlive: 10000,
  },
  deepseek: {
    modelName: MODEL_NAME,
    temperature: TEMPERATURE,
    apiKey: DEEPSEEK_API_KEY,
    maxTokens: 8192,
  },
  openai: {
    modelName: MODEL_NAME,
    temperature: TEMPERATURE,
    openAIApiKey: OPENAI_API_KEY,
    baseURL: OPENAI_BASE_URL,
    configuration: {
      baseURL: OPENAI_BASE_URL,
    },
    maxTokens: 8192,
  },
};

// Create model instance based on provider
function createModel() {
  if (MODEL_PROVIDER === "ollama") {
    if (!OLLAMA_BASE_URL) {
      throw new Error("OLLAMA_BASE_URL is required for Ollama model");
    }
    const model = new ChatOllama({
      ...modelConfigs.ollama
    });
    return model.bindTools(tools);
  } else if (MODEL_PROVIDER === "deepseek") {
    if (!DEEPSEEK_API_KEY) {
      throw new Error("DEEPSEEK_API_KEY is required for DeepSeek model");
    }
    const model = new ChatDeepSeek({
      ...modelConfigs.deepseek,
      streaming: false,
    });
    return model.bindTools(tools);
  } else if (MODEL_PROVIDER === "openai") {
    if (!OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY is required for OpenAI model");
    }
    const model = new ChatOpenAI({
      ...modelConfigs.openai,
      streaming: false,
    });
    return model.bindTools(tools);
  } else {
    throw new Error(`Unsupported model provider: ${MODEL_PROVIDER}`);
  }
}

// Create model instance
const model = createModel();

// Define the function that determines whether to continue or not
function shouldContinue(state: typeof StateAnnotation.State) {
  const messages = state.messages;
  const lastMessage = messages[messages.length - 1] as AIMessage;

  // Check if the last message has tool calls
  if (lastMessage.additional_kwargs?.tool_calls?.length > 0) {
    return "tools";
  }
  
  // If no tool calls and it's an AI message, end the conversation
  if (lastMessage instanceof AIMessage) {
    return "__end__";
  }
  
  // If it's a human message, continue to agent
  return "agent";
}

// Define the function that calls the model
async function callModel(state: typeof StateAnnotation.State) {
  const messages = state.messages;
  try {
    const response = await model.invoke(messages);
    if (response.additional_kwargs?.tool_calls) {
      console.log("Tool calls in response:", JSON.stringify(response.additional_kwargs.tool_calls, null, 2));
    }
    return { messages: [response] };
  } catch (error) {
    console.error("Error in callModel:", error);
    throw error;
  }
}

// Define the graph
const workflow = new StateGraph(StateAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge("__start__", "agent")
  .addConditionalEdges("agent", shouldContinue)
  .addEdge("tools", "agent");

// Initialize memory
const checkpointer = new MemorySaver();

// Compile the graph
const app = workflow.compile({ checkpointer });

// Example usage
async function main() {
  console.log("Starting conversation...");
  
  const finalState = await app.invoke(
    {
      messages: [
        new SystemMessage(systemPrompt),
        new HumanMessage(
          "介绍此项目，请使用工具搜索和读取相关文件，包括但不限于 README.md、package.json 等"
        ),
      ],
    },
    { configurable: { thread_id: "repo-agent-1" } }
  );

  // Log all messages for debugging
  console.log("\nConversation history:");
  finalState.messages.forEach((msg, i) => {
    console.log(`\nMessage ${i + 1}:`);
    console.log("Type:", msg.constructor.name);
    console.log("Content:", msg.content);
    if (msg instanceof AIMessage) {
      console.log("Additional kwargs:", JSON.stringify(msg.additional_kwargs, null, 2));
      if (msg.content === "{}") {
        console.warn("Warning: Empty response from model!");
      }
    }
  });
}

main().catch(error => {
  console.error("Error in main:", error);
  process.exit(1);
}); 