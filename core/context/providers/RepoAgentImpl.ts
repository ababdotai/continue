import { AIMessage, BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { ChatOllama } from "@langchain/ollama";
import { ChatDeepSeek } from "@langchain/deepseek";
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph } from "@langchain/langgraph";
import { MemorySaver, Annotation, messagesStateReducer } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import * as path from "node:path";
import { IDE } from "../../index.js";
import { z } from "zod";
import dotenv from "dotenv";

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

// Create tools using the provided IDE instance
function createTools(ide: IDE) {
  const searchCodeTool = tool(async (args) => {
    const { query } = z.object({ query: z.string() }).parse(args);
    try {
      const results = await ide.getSearchResults(query);
      return results;
    } catch (error: any) {
      return `Error searching code: ${error.message}`;
    }
  }, {
    name: "search_code",
    description: "Search for relevant code in the repository using semantic search.",
    schema: z.object({
      query: z.string().describe("The search query to find relevant code."),
    }),
  });

  const readFileTool = tool(async (args) => {
    const { filepath } = z.object({ filepath: z.string() }).parse(args);
    try {
      const content = await ide.readFile(path.join(WORKSPACE_ROOT, filepath));
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

  const editFileTool = tool(async (args) => {
    const { filepath, content } = z.object({ 
      filepath: z.string(),
      content: z.string()
    }).parse(args);
    try {
      await ide.writeFile(path.join(WORKSPACE_ROOT, filepath), content);
      return `Successfully edited file: ${filepath}`;
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

  const listDirTool = tool(async (args) => {
    const { dirpath } = z.object({ dirpath: z.string() }).parse(args);
    try {
      const entries = await ide.listDir(path.join(WORKSPACE_ROOT, dirpath));
      return JSON.stringify(entries.map(([name, type]) => ({
        name,
        type: type === 2 ? "directory" : "file"
      })), null, 2);
    } catch (error: any) {
      return `Error listing directory: ${error.message}`;
    }
  }, {
    name: "list_dir",
    description: "List the contents of a directory in the repository.",
    schema: z.object({
      dirpath: z.string().describe("The relative path to the directory from workspace root."),
    }),
  });

  const getProblems = tool(async (args) => {
    const { filepath } = z.object({ filepath: z.string().optional() }).parse(args);
    try {
      const problems = await ide.getProblems(filepath);
      return JSON.stringify(problems, null, 2);
    } catch (error: any) {
      return `Error getting problems: ${error.message}`;
    }
  }, {
    name: "get_problems",
    description: "Get diagnostic problems (errors, warnings) for a file.",
    schema: z.object({
      filepath: z.string().optional().describe("Optional file path to get problems for. If not provided, gets problems for the current file."),
    }),
  });

  return [searchCodeTool, readFileTool, editFileTool, listDirTool, getProblems];
}

// Create model instance based on provider
function createModel(tools: any[]) {
  let model;
  switch (MODEL_PROVIDER) {
    case "ollama":
      model = new ChatOllama({
        baseUrl: OLLAMA_BASE_URL,
        model: MODEL_NAME,
        temperature: TEMPERATURE,
      });
      break;
    case "deepseek":
      if (!DEEPSEEK_API_KEY) {
        throw new Error("DEEPSEEK_API_KEY is required for DeepSeek provider");
      }
      model = new ChatDeepSeek({
        apiKey: DEEPSEEK_API_KEY,
        model: MODEL_NAME,
        temperature: TEMPERATURE,
      });
      break;
    case "openai":
      if (!OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is required for OpenAI provider");
      }
      model = new ChatOpenAI({
        configuration: {
          apiKey: OPENAI_API_KEY,
          baseURL: OPENAI_BASE_URL,
        },
        openAIApiKey: OPENAI_API_KEY,
        model: MODEL_NAME,
        temperature: TEMPERATURE,
      });
      break;
    default:
      throw new Error(`Unsupported model provider: ${MODEL_PROVIDER}`);
  }
  return model.bindTools(tools);
}

// Define the function that determines whether to continue or not
function shouldContinue(state: typeof StateAnnotation.State) {
  const messages = state.messages;
  const lastMessage = messages[messages.length - 1] as AIMessage;

  // Check if the last message has tool calls
  if (lastMessage.additional_kwargs?.tool_calls && lastMessage.additional_kwargs.tool_calls.length > 0) {
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
function createCallModel(model: any) {
  return async function callModel(state: typeof StateAnnotation.State) {
    const messages = state.messages;
    try {
      const response = await model.invoke(messages);
      return { messages: [response] };
    } catch (error) {
      console.error("Error in callModel:", error);
      throw error;
    }
  };
}

// Define the system prompt
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

4. list_dir: List the contents of a directory
   - Use this to explore the repository structure
   - Helps in finding relevant files and directories

5. get_problems: Get diagnostic problems for files
   - Use this to check for errors and warnings
   - Helps in identifying issues that need attention

WORKFLOW GUIDELINES:
1. ALWAYS start by using search_code to find relevant information
2. Use list_dir to explore directories when needed
3. Use read_file to examine files found through search or directory listing
4. Never make assumptions without checking the code first
5. Provide explanations based on actual code, not assumptions
6. When asked about the project:
   - Search for README files
   - Look for package.json or similar config files
   - Search for main entry points
   - Examine project structure using list_dir

RESPONSE LANGUAGE:
- Match your response language to the user's query language
- Use English for code, comments, and technical terms
- Maintain consistent formatting and clear structure

IMPORTANT: You MUST use the tools to gather information before responding. Do not make assumptions or ask questions without first attempting to find the answers using the available tools.`;

// Function to initialize the repo agent
export async function initRepoAgent(ide: IDE) {
  const tools = createTools(ide);
  const toolNode = new ToolNode(tools);
  const model = createModel(tools);
  const callModel = createCallModel(model);

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

  return {
    async invoke(input: string) {
      console.log("Starting conversation...");
      
      const finalState = await app.invoke(
        {
          messages: [
            new SystemMessage(systemPrompt),
            new HumanMessage(input),
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

      return finalState;
    }
  };
}

// Add message handler for VSCode extension
export async function handleRepoAgentMessage(message: { type: string; payload: any }, ide: IDE) {
  if (message.type === "INVOKE_REPO_AGENT") {
    try {
      const agent = await initRepoAgent(ide);
      const result = await agent.invoke(message.payload.input);
      return {
        type: "REPO_AGENT_RESPONSE",
        payload: {
          messages: result.messages
        }
      };
    } catch (error: any) {
      return {
        type: "REPO_AGENT_ERROR",
        payload: {
          error: error.message
        }
      };
    }
  }
  return null;
} 