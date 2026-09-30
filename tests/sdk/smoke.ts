import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";

import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenAI } from "@google/genai";
import OpenAI from "openai";

const gatewayPort = Number(process.env.UNLLM_TEST_GATEWAY_PORT ?? "19080");
const upstreamPort = Number(process.env.UNLLM_TEST_UPSTREAM_PORT ?? "19081");
const gatewayBase = "http://127.0.0.1:" + gatewayPort;
const upstreamBase = "http://127.0.0.1:" + upstreamPort;

async function readJson(request: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) chunks.push(Buffer.from(chunk));
  return JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>;
}

const upstream = createServer(async (request: IncomingMessage, response: ServerResponse) => {
  if (request.url !== "/v1/responses" || request.method !== "POST") {
    response.writeHead(404, { "content-type": "application/json" });
    response.end(JSON.stringify({ error: { message: "not found" } }));
    return;
  }
  const body = await readJson(request);
  const tools = Array.isArray(body.tools) && body.tools.length > 0;
  if (body.stream === true) {
    response.writeHead(200, { "content-type": "text/event-stream" });
    response.write(
      "event: response.created\ndata: " +
        JSON.stringify({
          type: "response.created",
          response: { id: "resp_test", model: "upstream-test", status: "in_progress" },
        }) +
        "\n\n",
    );
    response.write(
      "event: response.output_text.delta\ndata: " +
        JSON.stringify({
          type: "response.output_text.delta",
          output_index: 0,
          content_index: 0,
          delta: "pong",
        }) +
        "\n\n",
    );
    response.end(
      "event: response.completed\ndata: " +
        JSON.stringify({
          type: "response.completed",
          response: { id: "resp_test", model: "upstream-test", status: "completed" },
        }) +
        "\n\n",
    );
    return;
  }
  response.writeHead(200, { "content-type": "application/json" });
  response.end(
    JSON.stringify({
      id: "resp_test",
      object: "response",
      model: "upstream-test",
      status: "completed",
      output: tools
        ? [
            {
              type: "function_call",
              id: "fc_test",
              call_id: "call_test",
              name: "echo",
              arguments: "{\"value\":\"pong\"}",
            },
          ]
        : [
            {
              type: "message",
              role: "assistant",
              content: [{ type: "output_text", text: "pong" }],
            },
          ],
      usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
    }),
  );
});

await new Promise<void>((resolve) => upstream.listen(upstreamPort, "127.0.0.1", resolve));

const directory = await mkdtemp(join(tmpdir(), "unllm-sdk-"));
const configPath = join(directory, "unllm.toml");
await writeFile(
  configPath,
  [
    "schema_version = 1",
    "",
    "[server]",
    'listen = "127.0.0.1:' + gatewayPort + '"',
    'log = "warn"',
    "",
    "[[upstreams]]",
    'name = "mock"',
    'protocol = "open_ai_responses"',
    'base_url = "' + upstreamBase + '/"',
    'token = "upstream-test"',
    "",
    "[[routes]]",
    'alias = "sdk-test"',
    'upstream = "mock"',
    'remote_model = "upstream-test"',
    'operations = ["generate"]',
    'generate_protocol = "open_ai_responses"',
    'mode = "strict"',
    'capabilities = ["streaming", "text", "function_tools"]',
    "",
  ].join("\n"),
  "utf8",
);

const binary =
  process.env.UNLLM_GATEWAY_BIN ?? join(process.cwd(), "target/debug/unllm-gateway");
const gateway = spawn(binary, ["serve", "--config", configPath], {
  stdio: ["ignore", "inherit", "inherit"],
});

async function waitUntilReady(): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      const response = await fetch(gatewayBase + "/healthz");
      if (response.ok) return;
    } catch {
      // The gateway may still be binding its socket.
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("Gateway did not become ready");
}

try {
  await waitUntilReady();
  const openai = new OpenAI({ apiKey: "client-test", baseURL: gatewayBase + "/v1" });
  const chat = await openai.chat.completions.create({
    model: "sdk-test",
    messages: [{ role: "user", content: "ping" }],
  });
  assert.equal(chat.choices[0]?.message.content, "pong");

  const response = await openai.responses.create({ model: "sdk-test", input: "ping" });
  assert.equal(response.output_text, "pong");

  const openaiTool = await openai.chat.completions.create({
    model: "sdk-test",
    messages: [{ role: "user", content: "Call echo." }],
    tools: [
      {
        type: "function",
        function: {
          name: "echo",
          parameters: { type: "object", properties: { value: { type: "string" } } },
        },
      },
    ],
  });
  const openaiToolCall = openaiTool.choices[0]?.message.tool_calls?.[0];
  assert.equal(openaiToolCall?.type, "function");
  if (openaiToolCall?.type === "function") {
    assert.equal(openaiToolCall.function.name, "echo");
  }

  const chatStream = await openai.chat.completions.create({
    model: "sdk-test",
    messages: [{ role: "user", content: "ping" }],
    stream: true,
  });
  let openaiStreamText = "";
  for await (const event of chatStream) {
    openaiStreamText += event.choices[0]?.delta.content ?? "";
  }
  assert.equal(openaiStreamText, "pong");

  const anthropic = new Anthropic({ apiKey: "client-test", baseURL: gatewayBase });
  const anthropicMessage = await anthropic.messages.create({
    model: "sdk-test",
    max_tokens: 64,
    messages: [{ role: "user", content: "ping" }],
  });
  assert.equal(anthropicMessage.content[0]?.type, "text");
  if (anthropicMessage.content[0]?.type === "text") {
    assert.equal(anthropicMessage.content[0].text, "pong");
  }

  const anthropicTool = await anthropic.messages.create({
    model: "sdk-test",
    max_tokens: 64,
    messages: [{ role: "user", content: "Call echo." }],
    tools: [
      {
        name: "echo",
        description: "Echo a value",
        input_schema: { type: "object", properties: { value: { type: "string" } } },
      },
    ],
  });
  assert.equal(anthropicTool.content[0]?.type, "tool_use");

  const anthropicStream = await anthropic.messages.create({
    model: "sdk-test",
    max_tokens: 64,
    messages: [{ role: "user", content: "ping" }],
    stream: true,
  });
  let sawAnthropicText = false;
  for await (const event of anthropicStream) {
    if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
      sawAnthropicText ||= event.delta.text === "pong";
    }
  }
  assert.equal(sawAnthropicText, true);

  const google = new GoogleGenAI({
    apiKey: "client-test",
    httpOptions: { baseUrl: gatewayBase },
  });
  const gemini = await google.models.generateContent({
    model: "sdk-test",
    contents: "ping",
  });
  assert.equal(gemini.text, "pong");

  const geminiTool = await google.models.generateContent({
    model: "sdk-test",
    contents: "Call echo.",
    config: {
      tools: [
        {
          functionDeclarations: [
            {
              name: "echo",
              parametersJsonSchema: {
                type: "object",
                properties: { value: { type: "string" } },
              },
            },
          ],
        },
      ],
    },
  });
  assert.equal(geminiTool.functionCalls?.[0]?.name, "echo");

  const geminiStream = await google.models.generateContentStream({
    model: "sdk-test",
    contents: "ping",
  });
  let geminiStreamText = "";
  for await (const event of geminiStream) geminiStreamText += event.text ?? "";
  assert.equal(geminiStreamText, "pong");

  await assert.rejects(() =>
    openai.responses.create({ model: "unknown-model", input: "ping" }),
  );
} finally {
  gateway.kill("SIGTERM");
  await once(gateway, "exit").catch(() => undefined);
  await new Promise<void>((resolve) => upstream.close(() => resolve()));
}
