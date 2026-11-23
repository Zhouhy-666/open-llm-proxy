import { describe, expect, it } from "vitest";
import {
  chatCompletionToTextCompletion,
  chatSseToTextCompletionSse,
  completionRequestToChat,
} from "~/src/requests/completions_v2";

function parseCompletionEvents(body: string): Record<string, unknown>[] {
  return body
    .split("\n\n")
    .filter(Boolean)
    .filter((block) => block !== "data: [DONE]")
    .map((block) => {
      const line = block
        .split("\n")
        .find((candidate) => candidate.startsWith("data: "));
      if (!line) throw new Error(`Missing data line: ${block}`);
      return JSON.parse(line.slice(6)) as Record<string, unknown>;
    });
}

describe("completionRequestToChat", () => {
  it("translates prompt, FIM suffix, and sampling options", () => {
    const chat = completionRequestToChat({
      model: "google-ai-studio/gemini-3.1-flash-lite",
      prompt: "def add(a, b):\n    ",
      suffix: "\n\nprint(add(1, 2))",
      max_tokens: 64,
      temperature: 0.2,
      top_p: 0.9,
      stop: ["\ndef "],
      stream: false,
    });

    expect(chat.model).toBe("google-ai-studio/gemini-3.1-flash-lite");
    expect(chat.messages).toEqual([
      {
        role: "user",
        content: "<PRE>def add(a, b):\n    <SUF>\n\nprint(add(1, 2))<MID>",
      },
    ]);
    expect(chat).toMatchObject({
      max_tokens: 64,
      temperature: 0.2,
      top_p: 0.9,
      stop: ["\ndef "],
      stream: false,
    });
  });

  it("does not double-wrap an already rendered FIM prompt", () => {
    const prompt = "<PRE>function f() {<SUF>}<MID>";
    const chat = completionRequestToChat({
      prompt,
      suffix: "}",
    });
    expect(chat.messages).toEqual([{ role: "user", content: prompt }]);
  });
});

describe("chatCompletionToTextCompletion", () => {
  it("returns the legacy text_completion shape", () => {
    const response = chatCompletionToTextCompletion(
      {
        id: "chatcmpl-1",
        created: 123,
        choices: [
          {
            message: { content: "return a + b" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      },
      {
        model: "google-ai-studio/gemini-3.1-flash-lite",
        prompt: "prefix",
      },
    );

    expect(response).toMatchObject({
      id: "chatcmpl-1",
      object: "text_completion",
      model: "google-ai-studio/gemini-3.1-flash-lite",
      choices: [
        {
          text: "return a + b",
          index: 0,
          finish_reason: "stop",
        },
      ],
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
      },
    });
  });
});

describe("chatSseToTextCompletionSse", () => {
  it("converts Chat Completions SSE into text completion SSE", async () => {
    const upstream = [
      'data: {"id":"chatcmpl-2","created":456,"choices":[{"delta":{"reasoning_content":"thinking"}}]}',
      "",
      'data: {"id":"chatcmpl-2","created":456,"choices":[{"delta":{"content":"return "}}]}',
      "",
      'data: {"id":"chatcmpl-2","created":456,"choices":[{"delta":{"content":"a + b"}}]}',
      "",
      'data: {"id":"chatcmpl-2","created":456,"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":4,"total_tokens":12}}',
      "",
      "data: [DONE]",
      "",
      "",
    ].join("\n");

    const body = new Response(upstream).body;
    if (!body) throw new Error("test response has no body");
    const converted = chatSseToTextCompletionSse(body, {
      model: "google-ai-studio/gemini-3.1-flash-lite",
    });
    const text = await new Response(converted).text();
    expect(text.endsWith("data: [DONE]\n\n")).toBe(true);

    const events = parseCompletionEvents(text);
    const content = events
      .flatMap((event) => event.choices as { text?: string }[])
      .map((choice) => choice.text ?? "")
      .join("");
    expect(content).toBe("return a + b");
    expect(events.some((event) => event.usage)).toBe(true);
    expect(events[0]).toMatchObject({
      object: "text_completion",
      model: "google-ai-studio/gemini-3.1-flash-lite",
    });
  });
});
