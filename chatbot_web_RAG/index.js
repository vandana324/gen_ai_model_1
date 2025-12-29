import dotenv from "dotenv";
import axios from "axios";
import * as cheerio from "cheerio";
import { ChromaClient } from "chromadb";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

// ✅ Initialize Gemini client
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// ✅ FIX 1: Changed model to 'gemini-pro'
// 'gemini-1.5-flash' was causing the 404 error.
// 'gemini-pro' is the standard stable model for generateContent.
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

// ✅ Initialize Chroma client
const chromaClient = new ChromaClient({
  host: "localhost",
  port: 8000,
  ssl: false,
});
const heartbeat = await chromaClient.heartbeat();
console.log("✅ Chroma heartbeat:", heartbeat);

const WebCollections = "WEB_SCAPED_DATA_COLLECTION-1";

// ------------------- SCRAPER -------------------
async function scrapeWebPage(url = "") {
  const { data } = await axios.get(url);
  const $ = cheerio.load(data);

  // Using .text() is fast but can be noisy.
  // For better results, you might target specific elements like 'article', 'main', or 'p'
  const pageHead = $("head").text();
  const pageBody = $("body")
    .clone() // Clone to avoid modifying the original
    .find("script, style, nav, footer, header") // Remove common noisy elements
    .remove()
    .end()
    .text();

  const internalLinks = new Set();
  const externalLinks = new Set();

  $("a").each((_, el) => {
    const link = $(el).attr("href");
    if (!link || link === "/") return;

    if (link.startsWith("http") || link.startsWith("https")) {
      externalLinks.add(link);
    } else {
      internalLinks.add(link);
    }
  });

  return {
    head: pageHead,
    body: pageBody.replace(/\s+/g, " ").trim(), // Clean up whitespace
    internalLinks: Array.from(internalLinks),
    externalLinks: Array.from(externalLinks),
  };
}

// ------------------- EMBEDDING (using Gemini) -------------------
async function generateVectorEmbeddings(text) {
  try {
    const embeddingModel = genAI.getGenerativeModel({
      model: "text-embedding-004",
    });

    const result = await embeddingModel.embedContent({
      content: {
        parts: [
          {
            text: typeof text === "string" ? text : JSON.stringify(text),
          },
        ],
      },
    });

    return result.embedding.values;
  } catch (error) {
    console.error("❌ Error generating embedding:", error.message);
    return [];
  }
}

// ------------------- INSERT INTO DB -------------------
// ✅ FIX 2: Added 'id' as a parameter.
async function insertIntoDB(id, embedding, url, body = "", head = "") {
  if (!embedding || embedding.length === 0) {
    console.warn("⚠️ Skipped empty embedding for:", id);
    return;
  }

  const collection = await chromaClient.getOrCreateCollection({
    name: WebCollections,
    embeddingFunction: null,
  });

  await collection.add({
    // ✅ FIX 2: Use the unique 'id' here instead of just 'url'
    ids: [id],
    embeddings: [embedding],
    metadatas: [{ url, head, body }],
  });
}

// ------------------- TEXT CHUNKER -------------------
function chunkText(text, chunkSize) {
  if (!text || chunkSize <= 0) throw new Error("Invalid input text");
  const words = text.split(/\s+/);
  const chunks = [];
  for (let i = 0; i < words.length; i += chunkSize) {
    chunks.push(words.slice(i, i + chunkSize).join(" "));
  }
  return chunks;
}

// ------------------- INGEST -------------------
async function ingest(url = "") {
  console.log(`🧠 Ingesting: ${url}`);
  try {
    const { head, body } = await scrapeWebPage(url);
    const bodyChunks = chunkText(body, 1000); // 1000 words is a large chunk, consider 250-500

    console.log(`... found ${bodyChunks.length} chunks`);

    // ✅ FIX 2: Loop with an index to create a unique ID for each chunk
    for (let i = 0; i < bodyChunks.length; i++) {
      const chunk = bodyChunks[i];
      const chunkId = `${url}-chunk-${i}`; // This is the new unique ID

      console.log(`... embedding chunk ${i + 1}/${bodyChunks.length}`);
      const bodyEmbedding = await generateVectorEmbeddings(chunk);

      // Pass the unique 'chunkId' to the database function
      await insertIntoDB(chunkId, bodyEmbedding, url, chunk, head);
    }

    console.log(`✅ Ingested: ${url}`);
  } catch (error) {
    console.error(`❌ Failed to ingest ${url}:`, error.message);
  }
}

// ------------------- CHAT QUERY -------------------
async function chat(question = "") {
  console.log("💬 Query:", question);

  const questionEmbedding = await generateVectorEmbeddings(question);
  const collection = await chromaClient.getOrCreateCollection({
    name: WebCollections,
    embeddingFunction: null,
  });

  const results = await collection.query({
    nResults: 3,
    queryEmbeddings: [questionEmbedding],
  });

  const metadatas = results?.metadatas?.[0] || [];
  if (metadatas.length === 0) {
    console.log("🤖: I'm sorry, I couldn't find any relevant context in my database to answer that.");
    return;
  }

  const Bodyresponse = metadatas.map((e) => e.body).filter(Boolean);
  const urlresponse = [...new Set(metadatas.map((e) => e.url))]; // Get unique URLs

  console.log("📄 Top URLs:", urlresponse);
  // console.log("📄 Top bodies:", Bodyresponse); // This can be very long, optional

  const prompt = `
You are an AI agent expert in providing support to users on behalf of a webpage.
Given the page content, reply accordingly. Answer the user's query based *only* on the retrieved context.
If the context is not sufficient to answer, say so.

Query: ${question}
URL(s): ${urlresponse.join(", ")}
Retrieved Context:
---
${Bodyresponse.join("\n---\n")}
---
Answer:`;

  const response = await model.generateContent(prompt);
  console.log("🤖:", response.response.text());
}

// ------------------- RUN -------------------
// Make sure ChromaDB is running locally on localhost:8000

// Uncomment this block to scrape and store data
// console.log("--- Starting Ingestion ---");
// await ingest("https://www.piyushgarg.dev");
// await ingest("https://www.piyushgarg.dev/cohort");
// await ingest("https://www.piyushgarg.dev/about");
// console.log("--- Ingestion Complete ---");

// Then run chat after ingestion
console.log("--- Starting Chat ---");
await chat("What are the things in cohort code ninja");