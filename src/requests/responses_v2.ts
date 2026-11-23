import type { ApiKeyAuth } from "../types";
import { newId, nowSeconds } from "../utils/crypto";
import { log } from "../utils/logger";
import { chatCompletionsV2 } from "./chat_completions_v2";

type JsonObject = Record<string, unknown>;

interface ResponsesInput {
  request: Request;
  env: Env;
  apiKeyAuth: ApiKeyAuth;
  ctx: { waitUntil(promise: Promise<unknown>): void };
}

interface ChatMessage extends JsonObject {
  role: string;
  content: string | JsonObject[] | null;
  tool_calls?: JsonObject[];
  tool_call_id?: string;
  name?: string;
}

interface ChatToolCall {
  id?: string;
  type?: string;
  index?: number;
  function?: {
    name?: string;
    arguments?: string;
  };
}

interface ChatCompletionChunk {
  id?: string;
  created?: number;
  model?: string;
  choices?: {
    index?: number;
    delta?: {
      role?: string;
      content?: string | null;
      reasoning_content?: string | null;
      tool_calls?: ChatToolCall[];
    };
    finish_reason?: string | null;
  }[];
  usage?: JsonObject;
  error?: {
    message?: string;
    type?: string;
    code?: string | null;
  };
}

interface ChatCompletion {
  id?: string;
  created?: number;
  model?: string;
  choices?: {
    message?: {
      role?: string;
      content?: string | JsonObject[] | null;
      tool_calls?: {
        id?: string;
        function?: { name?: string; arguments?: string };
      }[];
    };
    finish_reason?: string | null;
  }[];
  usage?: JsonObject;
}

interface ResponseUsage {
  input_tokens: number;
  input_tokens_details: { cached_tokens: number };
  output_tokens: number;
  output_tokens_details: { reasoning_tokens: number };
  total_tokens: number;
}

interface TextStreamState {
  id: string;
  outputIndex: number;
  text: string;
  added: boolean;
  done: boolean;
}

interface ToolStreamState {
  itemId: string;
  callId: string;
  name: string;
  arguments: string;
  outputIndex: number | null;
  added: boolean;
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function textValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

function responseError(message: string, status = 400): Response {
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

function convertContent(content: unknown, role: string): string | JsonObject[] {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return textValue(content);

  const parts = content.flatMap<JsonObject>((part) => {
    if (typeof part === "string") return [{ type: "text", text: part }];
    if (!isObject(part)) return [];

    const type = typeof part.type === "string" ? part.type : "";
    if (
      type === "input_text" ||
      type === "output_text" ||
      type === "text" ||
      type === "summary_text"
    ) {
      return [{ type: "text", text: textValue(part.text) }];
    }
    if (type === "refusal") {
      return [{ type: "text", text: textValue(part.refusal) }];
    }
    if (type === "input_image") {
      const image = part.image_url;
      const url =
        typeof image === "string"
          ? image
          : isObject(image)
            ? textValue(image.url)
            : "";
      if (!url) return [];
      return [
        {
          type: "image_url",
          image_url: {
            url,
            ...(typeof part.detail === "string" ? { detail: part.detail } : {}),
          },
        },
      ];
    }
    return [];
  });

  if (parts.length === 0) return "";
  if (parts.every((part) => part.type === "text")) {
    return parts.map((part) => textValue(part.text)).join("\n");
  }
  if (role === "assistant") {
    return parts
      .filter((part) => part.type === "text")
      .map((part) => textValue(part.text))
      .join("\n");
  }
  return parts;
}

function convertTools(tools: unknown): JsonObject[] | undefined {
  if (!Array.isArray(tools)) return undefined;
  const converted = tools.flatMap<JsonObject>((raw) => {
    if (!isObject(raw) || raw.type !== "function") return [];
    const nested = isObject(raw.function) ? raw.function : undefined;
    const name = textValue(raw.name ?? nested?.name).trim();
    if (!name) return [];
    const parameters = raw.parameters ??
      nested?.parameters ?? {
        type: "object",
        properties: {},
      };
    return [
      {
        type: "function",
        function: {
          name,
          ...((raw.description ?? nested?.description)
            ? {
                description: textValue(raw.description ?? nested?.description),
              }
            : {}),
          parameters,
        },
      },
    ];
  });
  return converted.length > 0 ? converted : undefined;
}

function convertToolChoice(choice: unknown): unknown {
  if (!isObject(choice) || choice.type !== "function") return choice;
  const nested = isObject(choice.function) ? choice.function : undefined;
  const name = textValue(choice.name ?? nested?.name).trim();
  if (!name) return undefined;
  return { type: "function", function: { name } };
}

function convertResponseFormat(value: unknown): unknown {
  if (!isObject(value)) return undefined;
  if (value.type === "json_object") return { type: "json_object" };
  if (value.type === "json_schema") {
    const schema = isObject(value.json_schema) ? value.json_schema : value;
    return {
      type: "json_schema",
      json_schema: {
        name: textValue(schema.name || "response"),
        schema: schema.schema ?? { type: "object" },
        ...(typeof schema.strict === "boolean"
          ? { strict: schema.strict }
          : {}),
      },
    };
  }
  return undefined;
}

/**
 * Translate an OpenAI Responses API request into the Chat Completions request
 * already supported by this proxy. Codex sends its full conversation history
 * in `input`, so no server-side response state is required for normal turns.
 */
export function responsesRequestToChat(body: JsonObject): JsonObject {
  const messages: ChatMessage[] = [];
  const toolNameById = new Map<string, string>();

  const pushAssistantToolCall = (item: JsonObject) => {
    const callId = textValue(item.call_id ?? item.id);
    const name = textValue(item.name).trim();
    const argumentsText = textValue(item.arguments || "{}");
    if (!callId || !name) return;
    toolNameById.set(callId, name);

    const toolCall = {
      id: callId,
      type: "function",
      function: { name, arguments: argumentsText },
    };
    const previous = messages[messages.length - 1];
    if (previous?.role === "assistant") {
      previous.tool_calls = [...(previous.tool_calls ?? []), toolCall];
      return;
    }
    messages.push({
      role: "assistant",
      content: null,
      tool_calls: [toolCall],
    });
  };

  const instructions = textValue(body.instructions).trim();
  if (instructions) messages.push({ role: "system", content: instructions });

  if (typeof body.input === "string") {
    messages.push({ role: "user", content: body.input });
  } else if (Array.isArray(body.input)) {
    for (const raw of body.input) {
      if (typeof raw === "string") {
        messages.push({ role: "user", content: raw });
        continue;
      }
      if (!isObject(raw)) continue;

      if (raw.type === "function_call") {
        pushAssistantToolCall(raw);
        continue;
      }
      if (raw.type === "function_call_output") {
        const callId = textValue(raw.call_id);
        if (!callId) continue;
        messages.push({
          role: "tool",
          tool_call_id: callId,
          ...(toolNameById.has(callId)
            ? { name: toolNameById.get(callId) }
            : {}),
          content:
            typeof raw.output === "string"
              ? raw.output
              : JSON.stringify(raw.output ?? ""),
        });
        continue;
      }
      if (raw.type === "message" || typeof raw.role === "string") {
        const rawRole = textValue(raw.role || "user");
        const role =
          rawRole === "developer"
            ? "system"
            : rawRole === "assistant"
              ? "assistant"
              : rawRole === "system"
                ? "system"
                : "user";
        const content = convertContent(raw.content, role);
        const previous = messages[messages.length - 1];
        if (
          role === "assistant" &&
          previous?.role === "assistant" &&
          typeof previous.content === "string" &&
          typeof content === "string"
        ) {
          previous.content = [previous.content, content]
            .filter(Boolean)
            .join("\n");
        } else {
          messages.push({ role, content });
        }
      }
    }
  }

  if (!messages.some((message) => message.role !== "system")) {
    messages.push({ role: "user", content: "" });
  }

  const tools = convertTools(body.tools);
  const toolChoice = convertToolChoice(body.tool_choice);
  const maxTokens =
    typeof body.max_output_tokens === "number"
      ? body.max_output_tokens
      : undefined;
  const text = isObject(body.text) ? body.text : undefined;
  const responseFormat = convertResponseFormat(text?.format);

  return {
    model: body.model,
    messages,
    stream: body.stream === true,
    ...(tools ? { tools } : {}),
    ...(toolChoice !== undefined ? { tool_choice: toolChoice } : {}),
    ...(typeof body.temperature === "number"
      ? { temperature: body.temperature }
      : {}),
    ...(typeof body.top_p === "number" ? { top_p: body.top_p } : {}),
    ...(maxTokens !== undefined ? { max_tokens: maxTokens } : {}),
    ...(responseFormat ? { response_format: responseFormat } : {}),
  };
}

function usageFromChat(raw: unknown): ResponseUsage {
  const usage = isObject(raw) ? raw : {};
  const inputDetails = isObject(usage.input_tokens_details)
    ? usage.input_tokens_details
    : isObject(usage.prompt_tokens_details)
      ? usage.prompt_tokens_details
      : {};
  const outputDetails = isObject(usage.output_tokens_details)
    ? usage.output_tokens_details
    : isObject(usage.completion_tokens_details)
      ? usage.completion_tokens_details
      : {};
  const input = Number(
    usage.input_tokens ?? usage.prompt_tokens ?? usage.inputTokens ?? 0,
  );
  const output = Number(
    usage.output_tokens ?? usage.completion_tokens ?? usage.outputTokens ?? 0,
  );
  return {
    input_tokens: Number.isFinite(input) ? input : 0,
    input_tokens_details: {
      cached_tokens: Number(
        inputDetails.cached_tokens ?? usage.prompt_cache_hit_tokens ?? 0,
      ),
    },
    output_tokens: Number.isFinite(output) ? output : 0,
    output_tokens_details: {
      reasoning_tokens: Number(outputDetails.reasoning_tokens ?? 0),
    },
    total_tokens: Number(
      usage.total_tokens ??
        usage.totalTokens ??
        (Number.isFinite(input) ? input : 0) +
          (Number.isFinite(output) ? output : 0),
    ),
  };
}

function messageText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .flatMap((part) => {
      if (!isObject(part)) return [];
      return typeof part.text === "string" ? [part.text] : [];
    })
    .join("");
}

function baseResponse(
  request: JsonObject,
  id: string,
  status: string,
  output: JsonObject[],
  usage?: ResponseUsage | null,
): JsonObject {
  const text = isObject(request.text) ? request.text : {};
  const include = Array.isArray(request.include) ? request.include : [];
  return {
    id,
    object: "response",
    created_at: nowSeconds(),
    status,
    background: false,
    error: null,
    incomplete_details:
      status === "incomplete" ? { reason: "max_output_tokens" } : null,
    instructions:
      typeof request.instructions === "string" ? request.instructions : null,
    max_output_tokens:
      typeof request.max_output_tokens === "number"
        ? request.max_output_tokens
        : null,
    max_tool_calls:
      typeof request.max_tool_calls === "number"
        ? request.max_tool_calls
        : null,
    model: typeof request.model === "string" ? request.model : "",
    output,
    parallel_tool_calls:
      typeof request.parallel_tool_calls === "boolean"
        ? request.parallel_tool_calls
        : true,
    previous_response_id:
      typeof request.previous_response_id === "string"
        ? request.previous_response_id
        : null,
    prompt_cache_key:
      typeof request.prompt_cache_key === "string"
        ? request.prompt_cache_key
        : null,
    reasoning: isObject(request.reasoning) ? request.reasoning : null,
    safety_identifier:
      typeof request.safety_identifier === "string"
        ? request.safety_identifier
        : null,
    service_tier:
      typeof request.service_tier === "string"
        ? request.service_tier
        : "default",
    store: request.store === true,
    temperature:
      typeof request.temperature === "number" ? request.temperature : null,
    text: {
      ...text,
      format:
        isObject(text.format) && typeof text.format.type === "string"
          ? text.format
          : { type: "text" },
    },
    tool_choice: request.tool_choice ?? "auto",
    tools: Array.isArray(request.tools) ? request.tools : [],
    top_logprobs:
      typeof request.top_logprobs === "number" ? request.top_logprobs : 0,
    top_p: typeof request.top_p === "number" ? request.top_p : 1,
    truncation:
      typeof request.truncation === "string" ? request.truncation : "disabled",
    usage: usage ?? null,
    user: typeof request.user === "string" ? request.user : null,
    metadata: isObject(request.metadata) ? request.metadata : {},
    ...(include.length > 0 ? { include } : {}),
  };
}

/** Convert a non-streaming Chat Completions response into Responses API JSON. */
export function chatCompletionToResponse(
  completion: ChatCompletion,
  request: JsonObject,
): JsonObject {
  const choice = completion.choices?.[0];
  const message = choice?.message;
  const output: JsonObject[] = [];
  const text = messageText(message?.content);

  if (text) {
    output.push({
      id: newId("msg"),
      type: "message",
      status: "completed",
      role: "assistant",
      content: [
        {
          type: "output_text",
          text,
          annotations: [],
          logprobs: [],
        },
      ],
    });
  }

  for (const toolCall of message?.tool_calls ?? []) {
    const callId = toolCall.id || newId("call");
    output.push({
      id: newId("fc"),
      type: "function_call",
      status: "completed",
      call_id: callId,
      name: textValue(toolCall.function?.name),
      arguments: textValue(toolCall.function?.arguments || "{}"),
    });
  }

  if (output.length === 0) {
    output.push({
      id: newId("msg"),
      type: "message",
      status: "completed",
      role: "assistant",
      content: [
        {
          type: "output_text",
          text: "",
          annotations: [],
          logprobs: [],
        },
      ],
    });
  }

  const status =
    choice?.finish_reason === "length" ? "incomplete" : "completed";
  return {
    ...baseResponse(
      request,
      newId("resp"),
      status,
      output,
      usageFromChat(completion.usage),
    ),
    output_text: text,
  };
}

function sse(type: string, data: JsonObject): string {
  return `event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`;
}

class ResponsesStreamConverter {
  private sequence = 0;
  private readonly responseId = newId("resp");
  private readonly createdAt = nowSeconds();
  private readonly output: JsonObject[] = [];
  private readonly tools = new Map<number, ToolStreamState>();
  private textState: TextStreamState | null = null;
  private usage: ResponseUsage | null = null;
  private finishReason: string | null = null;
  private failed: { message: string; code?: string | null } | null = null;

  constructor(private readonly request: JsonObject) {}

  private event(type: string, data: JsonObject): string {
    return sse(type, {
      ...data,
      sequence_number: this.sequence++,
    });
  }

  private currentResponse(status: string): JsonObject {
    return {
      ...baseResponse(
        this.request,
        this.responseId,
        status,
        this.output,
        this.usage,
      ),
      created_at: this.createdAt,
    };
  }

  private ensureText(): TextStreamState {
    if (this.textState) return this.textState;
    const state: TextStreamState = {
      id: newId("msg"),
      outputIndex: this.output.length,
      text: "",
      added: false,
      done: false,
    };
    this.textState = state;
    return state;
  }

  private addTextItem(state: TextStreamState): string {
    if (!state.added) {
      state.added = true;
      this.output.push({
        id: state.id,
        type: "message",
        status: "in_progress",
        role: "assistant",
        content: [],
      });
    }
    return this.event("response.output_item.added", {
      output_index: state.outputIndex,
      item: {
        id: state.id,
        type: "message",
        status: "in_progress",
        role: "assistant",
        content: [],
      },
    });
  }

  private addTextPart(state: TextStreamState): string {
    return this.event("response.content_part.added", {
      item_id: state.id,
      output_index: state.outputIndex,
      content_index: 0,
      part: {
        type: "output_text",
        text: "",
        annotations: [],
        logprobs: [],
      },
    });
  }

  responseCreated(): string[] {
    return [
      this.event("response.created", {
        response: this.currentResponse("in_progress"),
      }),
      this.event("response.in_progress", {
        response: this.currentResponse("in_progress"),
      }),
    ];
  }

  addTextDelta(delta: string): string[] {
    if (!delta) return [];
    const state = this.ensureText();
    const events: string[] = [];
    if (!state.added) {
      events.push(this.addTextItem(state), this.addTextPart(state));
    }
    state.text += delta;
    this.output[state.outputIndex] = {
      id: state.id,
      type: "message",
      status: "in_progress",
      role: "assistant",
      content: [
        {
          type: "output_text",
          text: state.text,
          annotations: [],
          logprobs: [],
        },
      ],
    };
    events.push(
      this.event("response.output_text.delta", {
        item_id: state.id,
        output_index: state.outputIndex,
        content_index: 0,
        delta,
        logprobs: [],
      }),
    );
    return events;
  }

  addToolDelta(tool: ChatToolCall): string[] {
    const index = tool.index ?? 0;
    let state = this.tools.get(index);
    if (!state) {
      state = {
        itemId: newId("fc"),
        callId: tool.id || newId("call"),
        name: tool.function?.name || "",
        arguments: "",
        outputIndex: null,
        added: false,
      };
      this.tools.set(index, state);
    }
    if (tool.id) state.callId = tool.id;
    if (tool.function?.name) state.name = tool.function.name;

    const delta = tool.function?.arguments ?? "";
    const events: string[] = [];
    if (!state.added) {
      state.added = true;
      state.outputIndex = this.output.length;
      this.output.push({
        id: state.itemId,
        type: "function_call",
        status: "in_progress",
        call_id: state.callId,
        name: state.name,
        arguments: "",
      });
      events.push(
        this.event("response.output_item.added", {
          output_index: state.outputIndex,
          item: {
            id: state.itemId,
            type: "function_call",
            status: "in_progress",
            call_id: state.callId,
            name: state.name,
            arguments: "",
          },
        }),
      );
    }
    if (delta) {
      state.arguments += delta;
      events.push(
        this.event("response.function_call_arguments.delta", {
          item_id: state.itemId,
          output_index: state.outputIndex,
          delta,
        }),
      );
    }
    this.output[state.outputIndex!] = {
      id: state.itemId,
      type: "function_call",
      status: "in_progress",
      call_id: state.callId,
      name: state.name,
      arguments: state.arguments,
    };
    return events;
  }

  consumeChunk(chunk: ChatCompletionChunk): string[] {
    if (chunk.usage) this.usage = usageFromChat(chunk.usage);
    if (chunk.error) {
      this.failed = {
        message: chunk.error.message || "Upstream provider error",
        code: chunk.error.code ?? chunk.error.type,
      };
      return [];
    }

    const choice = chunk.choices?.[0];
    if (!choice) return [];
    if (choice.finish_reason) this.finishReason = choice.finish_reason;

    const delta = choice.delta;
    if (!delta) return [];
    const events = this.addTextDelta(delta.content ?? "");
    for (const tool of delta.tool_calls ?? []) {
      events.push(...this.addToolDelta(tool));
    }
    return events;
  }

  fail(message: string): string[] {
    return [
      this.event("response.failed", {
        response: {
          ...this.currentResponse("failed"),
          error: {
            code: "upstream_error",
            message,
          },
        },
      }),
    ];
  }

  finish(): string[] {
    if (this.failed) return this.fail(this.failed.message);

    const events: string[] = [];
    if (!this.textState && this.tools.size === 0) this.ensureText();

    if (this.textState) {
      const state = this.textState;
      if (!state.added) {
        events.push(this.addTextItem(state), this.addTextPart(state));
      }
      const message = {
        id: state.id,
        type: "message",
        status: "completed",
        role: "assistant",
        content: [
          {
            type: "output_text",
            text: state.text,
            annotations: [],
            logprobs: [],
          },
        ],
      };
      events.push(
        this.event("response.output_text.done", {
          item_id: state.id,
          output_index: state.outputIndex,
          content_index: 0,
          text: state.text,
          logprobs: [],
        }),
        this.event("response.content_part.done", {
          item_id: state.id,
          output_index: state.outputIndex,
          content_index: 0,
          part: message.content[0],
        }),
        this.event("response.output_item.done", {
          output_index: state.outputIndex,
          item: message,
        }),
      );
      this.output[state.outputIndex] = message;
      state.done = true;
    }

    for (const state of this.tools.values()) {
      if (!state.added) {
        state.added = true;
        state.outputIndex = this.output.length;
        this.output.push({
          id: state.itemId,
          type: "function_call",
          status: "in_progress",
          call_id: state.callId,
          name: state.name || "tool",
          arguments: "",
        });
        events.push(
          this.event("response.output_item.added", {
            output_index: state.outputIndex,
            item: this.output[state.outputIndex],
          }),
        );
      }
      const item = {
        id: state.itemId,
        type: "function_call",
        status: "completed",
        call_id: state.callId,
        name: state.name || "tool",
        arguments: state.arguments || "{}",
      };
      events.push(
        this.event("response.function_call_arguments.done", {
          item_id: state.itemId,
          output_index: state.outputIndex,
          arguments: item.arguments,
        }),
        this.event("response.output_item.done", {
          output_index: state.outputIndex,
          item,
        }),
      );
      this.output[state.outputIndex!] = item;
    }

    const incomplete = this.finishReason === "length";
    const status = incomplete ? "incomplete" : "completed";
    events.push(
      this.event(incomplete ? "response.incomplete" : "response.completed", {
        response: this.currentResponse(status),
      }),
    );
    return events;
  }
}

/**
 * Convert the proxy's OpenAI Chat Completions SSE stream into Responses API
 * events. The converter is transport-agnostic so it can be tested directly.
 */
export function chatCompletionsSseToResponsesSse(
  upstream: ReadableStream<Uint8Array>,
  request: JsonObject,
): ReadableStream<Uint8Array> {
  const converter = new ResponsesStreamConverter(request);
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";
  let finished = false;
  let cancelled = false;
  let upstreamReader: ReadableStreamDefaultReader<Uint8Array> | undefined;

  return new ReadableStream<Uint8Array>({
    start(controller) {
      const emit = (events: string[]) => {
        if (cancelled) return;
        for (const event of events) controller.enqueue(encoder.encode(event));
      };
      const finish = () => {
        if (finished) return;
        finished = true;
        emit(converter.finish());
        if (!cancelled) controller.close();
      };

      emit(converter.responseCreated());

      void (async () => {
        const reader = upstream.getReader();
        upstreamReader = reader;
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
                finish();
                return;
              }
              if (data) {
                try {
                  emit(converter.consumeChunk(JSON.parse(data)));
                } catch {
                  log.debug("[responses] Ignored malformed upstream SSE chunk");
                }
              }
              boundary = buffer.indexOf("\n\n");
            }
          }
          finish();
        } catch (error) {
          if (!cancelled) {
            const message =
              error instanceof Error ? error.message : String(error);
            emit(converter.fail(message));
            finished = true;
            controller.close();
          }
        } finally {
          reader.releaseLock();
        }
      })();
    },
    cancel() {
      cancelled = true;
      void upstreamReader?.cancel().catch(() => undefined);
    },
  });
}

/** POST /responses and /v1/responses compatibility route. */
export async function responsesV2(input: ResponsesInput): Promise<Response> {
  const { request, env, apiKeyAuth, ctx } = input;
  let body: JsonObject;
  try {
    const parsed = await request.clone().json();
    if (!isObject(parsed)) return responseError("Invalid request body");
    body = parsed;
  } catch {
    return responseError("Invalid JSON request body");
  }

  const stream = body.stream === true;
  const chatBody = responsesRequestToChat(body);
  const headers = new Headers(request.headers);
  headers.set("content-type", "application/json");
  headers.delete("content-length");
  headers.delete("content-encoding");
  if (stream) headers.set("accept", "text/event-stream");

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

  if (stream) {
    if (!upstream.body) {
      return responseError("Upstream provider returned an empty stream", 502);
    }
    return new Response(chatCompletionsSseToResponsesSse(upstream.body, body), {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        ...(upstream.headers.get("x-cache")
          ? { "X-Cache": upstream.headers.get("x-cache")! }
          : {}),
      },
    });
  }

  let completion: ChatCompletion;
  try {
    completion = (await upstream.json()) as ChatCompletion;
  } catch {
    return responseError(
      "Upstream provider returned an invalid chat completion response",
      502,
    );
  }
  return new Response(
    JSON.stringify(chatCompletionToResponse(completion, body)),
    {
      headers: {
        "Content-Type": "application/json",
        ...(upstream.headers.get("x-cache")
          ? { "X-Cache": upstream.headers.get("x-cache")! }
          : {}),
      },
    },
  );
}
