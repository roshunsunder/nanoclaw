/**
 * Ollama Runner for NanoClaw
 * Handles LLM inference via a local Ollama instance as an alternative to
 * the Claude container-based agent. Maintains per-group conversation history.
 */
import fs from 'fs';
import path from 'path';

import { ASSISTANT_NAME, DATA_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL } from './config.js';
import { ContainerOutput } from './container-runner.js';
import { logger } from './logger.js';
import { RegisteredGroup } from './types.js';

interface OllamaMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface OllamaChatRequest {
  model: string;
  messages: OllamaMessage[];
  stream: boolean;
  options?: {
    temperature?: number;
    num_predict?: number;
  };
}

interface OllamaChatResponse {
  message: OllamaMessage;
  done: boolean;
}

function historyPath(groupFolder: string): string {
  return path.join(DATA_DIR, 'sessions', groupFolder, 'ollama-history.json');
}

function loadHistory(groupFolder: string): OllamaMessage[] {
  const file = historyPath(groupFolder);
  try {
    if (fs.existsSync(file)) {
      return JSON.parse(fs.readFileSync(file, 'utf-8'));
    }
  } catch {
    // Corrupted file — start fresh
  }
  return [];
}

function saveHistory(groupFolder: string, messages: OllamaMessage[]): void {
  const file = historyPath(groupFolder);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  // Keep last 100 messages to avoid unbounded growth
  const trimmed = messages.slice(-100);
  fs.writeFileSync(file, JSON.stringify(trimmed, null, 2));
}

function buildSystemPrompt(group: RegisteredGroup): string {
  return (
    `You are ${ASSISTANT_NAME}, a helpful personal assistant in a chat group called "${group.name}". ` +
    `Messages are provided in XML format with sender names and timestamps. ` +
    `Respond conversationally and helpfully. Keep responses concise unless detail is requested. ` +
    `Do not wrap your reply in XML tags.`
  );
}

async function callOllamaChat(
  model: string,
  messages: OllamaMessage[],
): Promise<string> {
  const url = `${OLLAMA_BASE_URL}/api/chat`;
  const body: OllamaChatRequest = {
    model,
    messages,
    stream: false,
  };

  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err) {
    throw new Error(
      `Cannot reach Ollama at ${OLLAMA_BASE_URL} — is it running? (${err instanceof Error ? err.message : String(err)})`,
    );
  }

  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new Error(`Ollama API error ${response.status}: ${text}`);
  }

  const data = (await response.json()) as OllamaChatResponse;
  return data.message?.content ?? '';
}

export async function runOllamaAgent(
  group: RegisteredGroup,
  prompt: string,
  model: string = OLLAMA_MODEL,
  onOutput?: (output: ContainerOutput) => Promise<void>,
): Promise<ContainerOutput> {
  const history = loadHistory(group.folder);

  // Prepend system prompt if conversation is fresh
  const messages: OllamaMessage[] = [];
  if (history.length === 0) {
    messages.push({ role: 'system', content: buildSystemPrompt(group) });
  } else {
    // Re-inject system prompt at the front of every call
    messages.push({ role: 'system', content: buildSystemPrompt(group) });
    messages.push(...history);
  }
  messages.push({ role: 'user', content: prompt });

  logger.info(
    { group: group.name, model, historyLength: history.length },
    'Calling Ollama',
  );

  try {
    const reply = await callOllamaChat(model, messages);

    // Persist history (without the leading system prompt — re-injected each call)
    history.push({ role: 'user', content: prompt });
    history.push({ role: 'assistant', content: reply });
    saveHistory(group.folder, history);

    const output: ContainerOutput = { status: 'success', result: reply };

    if (onOutput) {
      await onOutput(output);
      // Emit a completion marker (matches what container runner emits)
      await onOutput({ status: 'success', result: null });
    }

    return output;
  } catch (err) {
    logger.error({ group: group.name, err }, 'Ollama runner error');
    const output: ContainerOutput = {
      status: 'error',
      result: null,
      error: err instanceof Error ? err.message : String(err),
    };
    if (onOutput) await onOutput(output);
    return output;
  }
}

/** Clear conversation history for a group (e.g., on /reset). */
export function clearOllamaHistory(groupFolder: string): void {
  const file = historyPath(groupFolder);
  if (fs.existsSync(file)) fs.unlinkSync(file);
  logger.info({ groupFolder }, 'Ollama history cleared');
}
