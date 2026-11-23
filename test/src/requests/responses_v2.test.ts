import { describe, expect, it } from "vitest";
import {
  chatCompletionsSseToResponsesSse,
  chatCompletionToResponse,
  responsesRequestToChat,
} from "~/src/requests/responses_v2";

const request = {
  model: "deepseek/deepseek-v4-flash",
  instructions: "You are a coding agent.",
  input: [
    {
      type: "message",
      role: "user",
      content: [{ type: "input_text", text: "List the files." }],
    },
    {
      type: "function_call",
      call_id: "call_1",
      name: "shell",
      arguments: '{"cmd":"ls"}',
    },
    {
      type: "function_call_output",
      call_id: "call_1",
      output: "README.md",
    },
  ],
  tools: [
    {
      type: "function",
      name: "shell",
      description: "Run a shell command",
      parameters: {
        type: "object",
        properties: { cmd: { type: "string" } },
        required: ["cmd"],
      },
      strict: false,
    },
  ],
  tool_choice: "auto",
  parallel_tool_calls: true,
  max_output_tokens: 128,
  reasoning: { effort: "high" },
  text: { format: { type: "text" } },
  stream: false,
};

function parseSse(body: string): Record<string, unknown>[] {
  return body
    .split("\n\n")
    .filter(Boolean)
    .map((block) => {
      const data = block.split("\n").find((line) => line.startsWith("data: "));
      if (!data) throw new Error(`SSE event has no data line: ${block}`);
      return JSON.parse(data.slice(6)) as Record<string, unknown>;
    });
}

async function convertSse(upstream: string, input = request) {
  const stream = new Response(upstream).body;
  if (!stream) throw new Error("test response has no body");
  const converted = chatCompletionsSseToResponsesSse(stream, input);
  return parseSse(await new Response(converted).text());
}

describe("responsesRequestToChat", () => {
  it("translates messages, tool history, tools, and sampling options", () => {
    const chat = responsesRequestToChat(request);
    expect(chat.model).toBe("deepseek/deepseek-v4-flash");
    expect(chat.max_tokens).toBe(128);
    expect(chat.tools).toEqual([
      {
        type: "function",
        function: {
          name: "shell",
          description: "Run a shell command",
          parameters: {
            type: "object",
            properties: { cmd: { type: "string" } },
            required: ["cmd"],
          },
        },
      },
    ]);

    expect(chat.messages).toEqual([
      { role: "system", content: "You are a coding agent." },
      { role: "user", content: "List the files." },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "shell", arguments: '{"cmd":"ls"}' },
          },
        ],
      },
      {
        role: "tool",
        tool_call_id: "call_1",
        name: "shell",
        content: "README.md",
      },
    ]);
  });

  it("accepts a plain string input", () => {
    const chat = responsesRequestToChat({ model: "m", input: "hello" });
    expect(chat.messages).toEqual([{ role: "user", content: "hello" }]);
  });
});

describe("chatCompletionToResponse", () => {
  it("converts text, function calls, usage, and finish status", () => {
    const response = chatCompletionToResponse(
      {
        id: "chatcmpl-1",
        created: 1,
        model: "deepseek-v4-flash",
        choices: [
          {
            finish_reason: "tool_calls",
            message: {
              role: "assistant",
              content: "Checking.",
              tool_calls: [
                {
                  id: "call_2",
                  function: {
                    name: "shell",
                    arguments: '{"cmd":"pwd"}',
                  },
                },
              ],
            },
          },
        ],
        usage: {
          prompt_tokens: 12,
          completion_tokens: 4,
          total_tokens: 16,
          prompt_tokens_details: { cached_tokens: 3 },
        },
      },
      request,
    );

    const output = response.output as Record<string, unknown>[];
    expect(response.object).toBe("response");
    expect(response.status).toBe("completed");
    expect(response.output_text).toBe("Checking.");
    expect(output).toHaveLength(2);
    expect(output[0]).toMatchObject({
      type: "message",
      role: "assistant",
      content: [{ type: "output_text", text: "Checking." }],
    });
    expect(output[1]).toMatchObject({
      type: "function_call",
      call_id: "call_2",
      name: "shell",
      arguments: '{"cmd":"pwd"}',
    });
    expect(response.usage).toMatchObject({
      input_tokens: 12,
      output_tokens: 4,
      total_tokens: 16,
      input_tokens_details: { cached_tokens: 3 },
    });
  });

  it("marks length-limited replies as incomplete", () => {
    const response = chatCompletionToResponse(
      {
        choices: [
          {
            finish_reason: "length",
            message: { role: "assistant", content: "Partial" },
          },
        ],
      },
      request,
    );
    expect(response.status).toBe("incomplete");
    expect(response.incomplete_details).toEqual({
      reason: "max_output_tokens",
    });
  });
});

describe("chatCompletionsSseToResponsesSse", () => {
  it("emits text and function-call response events", async () => {
    const upstream = [
      'data: {"choices":[{"delta":{"role":"assistant"}}]}',
      "",
      'data: {"choices":[{"delta":{"content":"Hello"}}]}',
      "",
      'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_3","type":"function","function":{"name":"shell","arguments":"{\\"cmd\\":"}}]}}]}',
      "",
      'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"pwd\\"}"}}]}}]}',
      "",
      'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}',
      "",
      "data: [DONE]",
      "",
      "",
    ].join("\n");

    const events = await convertSse(upstream);
    const types = events.map((event) => event.type);
    expect(types[0]).toBe("response.created");
    expect(types).toContain("response.output_text.delta");
    expect(types).toContain("response.function_call_arguments.delta");
    expect(types[types.length - 1]).toBe("response.completed");

    const text = events
      .filter((event) => event.type === "response.output_text.delta")
      .map((event) => event.delta)
      .join("");
    expect(text).toBe("Hello");

    const argumentDeltas = events
      .filter(
        (event) => event.type === "response.function_call_arguments.delta",
      )
      .map((event) => event.delta)
      .join("");
    expect(argumentDeltas).toBe('{"cmd":"pwd"}');

    const argsDone = events.find(
      (event) => event.type === "response.function_call_arguments.done",
    );
    expect(argsDone?.arguments).toBe('{"cmd":"pwd"}');

    const completed = events[events.length - 1];
    const response = completed?.response as Record<string, unknown>;
    expect(response.status).toBe("completed");
    expect(response.output).toHaveLength(2);
    expect(response.usage).toMatchObject({
      input_tokens: 7,
      output_tokens: 3,
      total_tokens: 10,
    });
  });
});
