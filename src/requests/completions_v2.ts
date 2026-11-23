import type { ApiKeyAuth } from "../types";
import { newId, nowSeconds } from "../utils/crypto";
import { chatCompletionsV2 } from "./chat_completions_v2";

type JsonObject = Record<string, unknown>;

interface CompletionsInput {
  request: Request;
  env: Env;
  apiKeyAuth: ApiKeyAuth;
  ctx: { waitUntil(promise: Promise<unknown>): void };
}

interface ChatCompletion {
  id?: string;
  created?: number;
  model?: string;
  choices?: {
    message?: {
      content?: string | { text?: string }[] | null;
    };
    finish_reason?: string | null;
  }[];
  usage?: JsonObject;
}

interface ChatCompletionChunk {
  id?: string;
  created?: number;
  model?: string;
  choices?: {
    delta?: {
      content?: string | null;
      reasoning_content?: string | null;
    };
    finish_reason?: string | null;
  }[];
  usage?: JsonObject;
  error?: { message?: string };
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function errorResponse(message: string, status = 400): Response {
  return new Response(
    JSON.stringify({
      error: {
        message,
        type: "invalid_request_error",
        param: null,
        code: "invalid_request",
      },
    }),
    {
      status,
      headers: { "Content-Type": "application/json" },
    },
  );
}

function promptText(prompt: unknown): string {
  if (typeof prompt === "string") return prompt;
  if (
    Array.isArray(prompt) &&
    prompt.every((part) => typeof part === "string")
  ) {
    return prompt.join("\n");
  }
  return "";
}

function withSuffix(prompt: string, suffix: unknown): string {
  if (typeof suffix !== "string" || !suffix) return prompt;
  // Common FIM markers. If the autocomplete plugin already rendered a FIM
  // template into `prompt`, keep it untouched.
  if (/<PRE>|<SUF>|<MID>|[\[<]SUFFIX[\]>]/i.test(prompt)) return prompt;
  return `<PRE>${prompt}<SUF>${suffix}<MID>`;
}

/** Translate the legacy OpenAI text-completions body into Chat Completions. */
export function completionRequestToChat(body: JsonObject): JsonObject {
  const prompt = withSuffix(promptText(body.prompt), body.suffix);
  return {
    model: body.model,
    messages: [{ role: "user", content: prompt }],
    stream: body.stream === true,
    ...(typeof body.max_tokens === "number"
      ? { max_tokens: body.max_tokens }
      : {}),
    ...(typeof body.temperature === "number"
      ? { temperature: body.temperature }
      : {}),
    ...(typeof body.top_p === "number" ? { top_p: body.top_p } : {}),
    ...(typeof body.presence_penalty === "number"
      ? { presence_penalty: body.presence_penalty }
      : {}),
    ...(typeof body.frequency_penalty === "number"
      ? { frequency_penalty: body.frequency_penalty }
      : {}),
    ...(typeof body.stop === "string" || Array.isArray(body.stop)
      ? { stop: body.stop }
      : {}),
  };
}

function completionText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .flatMap((part) => {
      if (!isObject(part) || typeof part.text !== "string") return [];
      return [part.text];
    })
    .join("");
}

/** Convert a Chat Completions response into the legacy text-completion shape. */
export function chatCompletionToTextCompletion(
  completion: ChatCompletion,
  request: JsonObject,
): JsonObject {
  const choice = completion.choices?.[0];
  const content = completionText(choice?.message?.content);
  const text =
    request.echo === true ? promptText(request.prompt) + content : content;
  return {
    id: completion.id ?? newId("cmpl"),
    object: "text_completion",
    created: completion.created ?? nowSeconds(),
    model:
      typeof request.model === "string"
        ? request.model
        : (completion.model ?? ""),
    choices: [
      {
        text,
        index: 0,
        logprobs: null,
        finish_reason: choice?.finish_reason ?? "stop",
      },
    ],
    usage: completion.usage ?? null,
  };
}

function textCompletionChunk(
  id: string,
  created: number,
  model: string,
  text: string,
  finishReason: string | null,
  usage?: JsonObject,
): string {
  const payload = {
    id,
    object: "text_completion",
    created,
    model,
    choices: [
      {
        text,
        index: 0,
        logprobs: null,
        finish_reason: finishReason,
      },
    ],
    ...(usage ? { usage } : {}),
  };
  return `data: ${JSON.stringify(payload)}\n\n`;
}

/** Translate Chat Completions SSE into legacy text-completion SSE. */
export function chatSseToTextCompletionSse(
  upstream: ReadableStream<Uint8Array>,
  request: JsonObject,
): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";
  let finished = false;
  let cancelled = false;
  let heartbeat: ReturnType<typeof setInterval> | undefined;
  let id = newId("cmpl");
  let created = nowSeconds();
  let model = typeof request.model === "string" ? request.model : "";
  let sentDone = false;

  return new ReadableStream<Uint8Array>({
    start(controller) {
      const clearHeartbeat = () => {
        if (heartbeat !== undefined) {
          clearInterval(heartbeat);
          heartbeat = undefined;
        }
      };
      const enqueue = (value: string) => {
        if (!cancelled) controller.enqueue(encoder.encode(value));
      };
      const finish = () => {
        if (finished) return;
        finished = true;
        clearHeartbeat();
        if (!sentDone) enqueue("data: [DONE]\n\n");
        if (!cancelled) controller.close();
      };
      const consume = (chunk: ChatCompletionChunk) => {
        if (chunk.error) {
          throw new Error(chunk.error.message || "Upstream provider error");
        }
        if (chunk.id) id = chunk.id;
        if (chunk.created) created = chunk.created;
        if (chunk.model) model = chunk.model;
        const choice = chunk.choices?.[0];
        const content = choice?.delta?.content ?? "";
        if (content) {
          enqueue(textCompletionChunk(id, created, model, content, null));
        } else if (choice?.delta?.reasoning_content) {
          // Keep completion clients active while the model is thinking.
          enqueue(textCompletionChunk(id, created, model, "", null));
        }
        if (choice?.finish_reason) {
          enqueue(
            textCompletionChunk(
              id,
              created,
              model,
              "",
              choice.finish_reason,
              chunk.usage,
            ),
          );
        }
      };

      heartbeat = setInterval(() => {
        if (!cancelled && !finished) {
          enqueue(textCompletionChunk(id, created, model, "", null));
        }
      }, 10_000);

      void (async () => {
        const reader = upstream.getReader();
        try {
          while (!cancelled) {
            const { value, done } = await reader.read();
            if (done) break;
            if (value) buffer += decoder.decode(value, { stream: true });
            buffer = buffer.replace(/\r\n/g, "\n");
            let boundary = buffer.indexOf("\n\n");
            while (boundary !== -1) {
              const block = buffer.slice(0, boundary);
              buffer = buffer.slice(boundary + 2);
              const data = block
                .split("\n")
                .filter((line) => line.startsWith("data:"))
                .map((line) => line.slice(5).trimStart())
                .join("\n")
                .trim();
              if (data === "[DONE]") {
                enqueue("data: [DONE]\n\n");
                sentDone = true;
                finish();
                return;
              }
              if (data) consume(JSON.parse(data) as ChatCompletionChunk);
              boundary = buffer.indexOf("\n\n");
            }
          }
          finish();
        } catch (error) {
          if (!cancelled) {
            clearHeartbeat();
            finished = true;
            controller.error(
              error instanceof Error ? error : new Error(String(error)),
            );
          }
        } finally {
          reader.releaseLock();
        }
      })();
    },
    cancel() {
      cancelled = true;
      if (heartbeat !== undefined) {
        clearInterval(heartbeat);
        heartbeat = undefined;
      }
    },
  });
}

/** POST /completions and /v1/completions compatibility route. */
export async function completionsV2(
  input: CompletionsInput,
): Promise<Response> {
  const { request, env, apiKeyAuth, ctx } = input;
  let body: JsonObject;
  try {
    const parsed = await request.clone().json();
    if (!isObject(parsed)) return errorResponse("Invalid request body");
    body = parsed;
  } catch {
    return errorResponse("Invalid JSON request body");
  }
  if (!promptText(body.prompt)) {
    return errorResponse("prompt must be a string or an array of strings");
  }

  const chatBody = completionRequestToChat(body);
  const headers = new Headers(request.headers);
  headers.set("content-type", "application/json");
  headers.delete("content-length");
  headers.delete("content-encoding");
  if (body.stream === true) headers.set("accept", "text/event-stream");

  const chatRequest = new Request(request.url, {
    method: "POST",
    headers,
    body: JSON.stringify(chatBody),
    signal: request.signal,
  });
  const upstream = await chatCompletionsV2({
    request: chatRequest,
    env,
    apiKeyAuth,
    ctx,
  });
  if (!upstream.ok) return upstream;

  if (body.stream === true) {
    if (!upstream.body) {
      return errorResponse("Upstream provider returned an empty stream", 502);
    }
    return new Response(chatSseToTextCompletionSse(upstream.body, body), {
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
      },
    });
  }

  let completion: ChatCompletion;
  try {
    completion = (await upstream.json()) as ChatCompletion;
  } catch {
    return errorResponse(
      "Upstream provider returned an invalid chat completion response",
      502,
    );
  }
  return new Response(
    JSON.stringify(chatCompletionToTextCompletion(completion, body)),
    {
      headers: { "Content-Type": "application/json" },
    },
  );
}
