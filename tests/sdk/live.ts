import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenAI } from "@google/genai";
import OpenAI from "openai";

const model = process.env.UNLLM_LIVE_MODEL;
const baseURL = process.env.UNLLM_LIVE_BASE_URL;
const apiKey = process.env.UNLLM_LIVE_API_KEY;

if (!model || !baseURL || !apiKey) {
  console.log("Live contract tests skipped because their secrets are not configured.");
  process.exit(0);
}

const openai = new OpenAI({ apiKey, baseURL: baseURL + "/v1" });
await openai.responses.create({ model, input: "Reply with OK." });

const anthropic = new Anthropic({ apiKey, baseURL });
await anthropic.messages.create({
  model,
  max_tokens: 16,
  messages: [{ role: "user", content: "Reply with OK." }],
});

const google = new GoogleGenAI({ apiKey, httpOptions: { baseUrl: baseURL } });
await google.models.generateContent({ model, contents: "Reply with OK." });
