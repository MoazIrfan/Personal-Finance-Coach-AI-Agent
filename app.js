import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { createOpenAIFunctionsAgent, AgentExecutor } from 'langchain/agents';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { DynamicTool } from '@langchain/core/tools';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Document } from '@langchain/core/documents';
import * as fs from 'fs/promises';
import * as dotenv from 'dotenv';
import readline from 'readline';

dotenv.config();


// Read and split the transactions into chunks
async function loadTransactionChunks() {
  const content = await fs.readFile('documents/transactions.csv', 'utf8');
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 100,
  });
  const documents = await splitter.createDocuments([content]);
  return documents;
}

// Create vector store from transaction chunks
async function createVectorStore(documents) {
  const embeddings = new OpenAIEmbeddings();
  const vectorStore = await MemoryVectorStore.fromDocuments(documents, embeddings);
  return vectorStore;
}

// Define the TransactionRetriever tool
function createTransactionRetrieverTool(vectorStore) {
  return new DynamicTool({
    name: 'TransactionRetriever',
    description:
      'Use this tool ONCE to fetch relevant transactions, amounts, categories, and dates from the user’s spending records. Use it to summarize spending by category or date range.',
    func: async (input) => {
      const results = await vectorStore.similaritySearch(input, 5);
      return results.map((doc) => doc.pageContent).join('\n---\n');
    },
  });
}

// Set up the Chat model
const model = new ChatOpenAI({
  modelName: 'gpt-4o',
  temperature: 0.2,
  openAIApiKey: process.env.OPENAI_API_KEY,
});

// Define the prompt template
const prompt = ChatPromptTemplate.fromMessages([
  [
    'system',
    'You are a helpful personal finance coach. Answer their questions about expenses by using the TransactionRetriever tool ONCE to summarize spending. Do not guess.',
  ],
  new MessagesPlaceholder('agent_scratchpad'),
  ['user', '{input}'],
]);

// Main execution function
async function main() {
  const documents = await loadTransactionChunks();
  const vectorStore = await createVectorStore(documents);
  const transactionRetrieverTool = createTransactionRetrieverTool(vectorStore);

  const agent = await createOpenAIFunctionsAgent({
    llm: model,
    tools: [transactionRetrieverTool],
    prompt,
  });

  const executor = new AgentExecutor({
    agent,
    tools: [transactionRetrieverTool],
    verbose: true,
    maxIterations: 10,
  });

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const askQuestion = (query) => {
    return new Promise((resolve) => {
      rl.question(query, resolve);
    });
  };

  console.log('\n💬 Ask me anything about your transactions. Type "exit" to quit.\n');

  while (true) {
    const userPrompt = await askQuestion('You: ');
    if (userPrompt.toLowerCase() === 'exit') {
      console.log('👋 Goodbye!');
      break;
    }

    const result = await executor.invoke({ input: userPrompt });
    console.log('\n💸 Spending Summary 💸\n');
    console.log(result.output + '\n');
  }

  rl.close();
}

main();
