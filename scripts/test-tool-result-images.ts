import assert from "node:assert/strict";
import type {
  Api,
  Context,
  ImageContent,
  Model,
  ToolCall,
  ToolResultMessage,
} from "@earendil-works/pi-ai";
import { buildRequest, convertMessages, friendlyAntigravityError } from "../src/stream/index.js";
import type { GeminiContent } from "../src/types/types.js";

const model = {
  id: "claude-opus-5-5",
  name: "Claude",
  api: "antigravity-api",
  provider: "antigravity",
  reasoning: true,
  input: ["text", "image"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 200000,
  maxTokens: 64000,
} as Model<Api>;
const runtime = "claude-opus-5-5-medium";
const calls: ToolCall[] = [
  { type: "toolCall", id: "call-bash", name: "bash", arguments: { command: "echo ok" } },
  { type: "toolCall", id: "call-dark", name: "read", arguments: { path: "dark.png" } },
  { type: "toolCall", id: "call/light", name: "read", arguments: { path: "light.png" } },
];
const ids = ["call-bash", "call-dark", "call_light"];
const images: ImageContent[] = ["dark", "light"].map((name) => ({
  type: "image",
  mimeType: "image/png",
  data: Buffer.from(name).toString("base64"),
}));
const zeroUsage = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function fixture(modelId = model.id): Context {
  return {
    messages: [
      { role: "user", content: "Compare the screenshots.", timestamp: 1 },
      {
        role: "assistant",
        content: calls.map((call) => ({ ...call, thoughtSignature: "YWJjZA==" })),
        api: "antigravity-api",
        provider: "antigravity",
        model: modelId,
        usage: zeroUsage,
        stopReason: "toolUse",
        timestamp: 2,
      },
      ...calls.map((call, index): ToolResultMessage => ({
        role: "toolResult",
        toolCallId: call.id,
        toolName: call.name,
        content: [
          { type: "text", text: index === 0 ? "ok" : "Read image file [image/png]" },
          ...(index === 0 ? [] : [images[index - 1]]),
        ],
        isError: false,
        timestamp: 3,
      })),
    ],
  };
}

function assertResultsFirst(turn: GeminiContent, expectedIds = ids): void {
  assert.equal(turn.role, "user");
  assert.deepEqual(
    turn.parts
      .slice(0, expectedIds.length)
      .map((part) => ("functionResponse" in part ? part.functionResponse.id : "not a tool result")),
    expectedIds,
    "all matching results must precede images and ordinary text",
  );
  assert.ok(turn.parts.slice(expectedIds.length).every((part) => !("functionResponse" in part)));
}

// Reproduce bash + two image reads from one assistant response. The old converter
// put the second read result after the first image, causing Claude's HTTP 400.
const context = fixture();
const original = structuredClone(context);
const converted = convertMessages(model, context, runtime);
assert.equal(converted.length, 3);
assertResultsFirst(converted[2]);
assert.deepEqual(
  converted[2].parts.filter((part) => "inlineData" in part).map((part) => part.inlineData),
  images.map(({ type: _, ...image }) => image),
  "both image payloads must survive in their original order",
);
for (const [index, part] of converted[2].parts.entries()) {
  if (!("inlineData" in part)) continue;
  const imageIndex = images.findIndex((image) => image.data === part.inlineData.data);
  assert.deepEqual(
    converted[2].parts[index - 1],
    {
      text: `Images from tool read (${ids[imageIndex + 1]}):`,
    },
    "deferred images must retain their source tool-call identity",
  );
}
assert.deepEqual(context, original, "conversion must not mutate saved history");

// Check the serialized request too, including runtime-only Claude detection.
for (const publicId of [model.id, "custom-model-alias"]) {
  const request = buildRequest({ ...model, id: publicId }, context, "test-project", {}, runtime);
  assertResultsFirst(JSON.parse(JSON.stringify(request)).request.contents[2]);
}

// A later result can be text-only or an error, and result arrival order may differ
// from call order. None may remain behind an image or an interleaved user message.
const mixed = fixture();
mixed.messages.splice(3, 0, { role: "user", content: "Check both themes.", timestamp: 3 });
const lastResult = mixed.messages.at(-1)!;
assert.ok(lastResult.role === "toolResult");
lastResult.isError = true;
lastResult.content = [{ type: "text", text: "File not found" }];
const mixedTurn = convertMessages(model, mixed, runtime).at(-1)!;
assertResultsFirst(mixedTurn);
assert.deepEqual(mixedTurn.parts[2], {
  functionResponse: { id: ids[2], name: "read", response: { error: "File not found" } },
});
assert.ok(mixedTurn.parts.some((part) => "text" in part && part.text === "Check both themes."));
const reversed = fixture();
reversed.messages.splice(2, 3, ...reversed.messages.slice(2).reverse());
assertResultsFirst(convertMessages(model, reversed, runtime).at(-1)!, [...ids].reverse());

// Multiple images from one call stay together and labelled; a later assistant
// turn must retain its own results instead of merging separate tool batches.
const multiple = fixture();
const firstRead = multiple.messages[3];
assert.ok(firstRead.role === "toolResult");
firstRead.content.push(images[1]);
const multiTurn = convertMessages(model, multiple, runtime).at(-1)!;
assertResultsFirst(multiTurn);
assert.equal(multiTurn.parts.filter((part) => "inlineData" in part).length, 3);
const history = fixture();
const next = fixture();
const firstAssistant = history.messages[1];
assert.ok(firstAssistant.role === "assistant");
history.messages.push(
  {
    ...firstAssistant,
    content: [{ type: "text", text: "First comparison complete." }],
    stopReason: "stop",
  },
  ...next.messages,
);
const historicalTurns = convertMessages(model, history, runtime).filter((turn) =>
  turn.parts.some((part) => "functionResponse" in part),
);
assert.equal(historicalTurns.length, 2);
historicalTurns.forEach((turn) => assertResultsFirst(turn));

// Ordinary tool responses remain unchanged. Gemini keeps its existing signed
// multimodal format rather than receiving Claude-only reordering or labels.
const textOnly = fixture();
for (const message of textOnly.messages) {
  if (message.role === "toolResult") message.content = [{ type: "text", text: "ok" }];
}
assert.equal(convertMessages(model, textOnly, runtime).at(-1)!.parts.length, 3);
const gemini = { ...model, id: "gemini-3.8-flash" };
const geminiTurn = convertMessages(gemini, fixture(gemini.id), "gemini-3.8-flash-low").at(-1)!;
assert.deepEqual(
  geminiTurn.parts.map((part) => Object.keys(part)[0]),
  ["functionResponse", "functionResponse", "inlineData", "functionResponse", "inlineData"],
);

const error = friendlyAntigravityError(
  400,
  JSON.stringify({
    type: "error",
    error: {
      type: "invalid_request_error",
      message:
        "messages.74: `tool_use` ids were found without `tool_result` blocks immediately after: call_light",
    },
  }),
);
assert.match(error, /tool.*result/i);
assert.doesNotMatch(error, /\/login|retry once/i);
assert.match(friendlyAntigravityError(401, "Unauthorized"), /\/login antigravity/);

console.log("tool-result-images: ok");
