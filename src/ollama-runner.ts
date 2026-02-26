/**
 * Ollama Runner for NanoClaw
 * Handles LLM inference via a local Ollama instance with tool-calling support.
 * Maintains per-group conversation history.
 */
import fs from 'fs';
import os from 'os';
import path from 'path';

import { google, gmail_v1 } from 'googleapis';

import { ASSISTANT_NAME, DATA_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_NUM_CTX } from './config.js';
import { ContainerOutput } from './container-runner.js';
import { logger } from './logger.js';
import { RegisteredGroup } from './types.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ToolCall {
  function: {
    name: string;
    arguments: Record<string, unknown>;
  };
}

interface OllamaMessage {
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  tool_calls?: ToolCall[];
}

interface OllamaToolFunction {
  name: string;
  description: string;
  parameters: {
    type: 'object';
    properties: Record<string, { type: string; description?: string; enum?: string[] }>;
    required?: string[];
  };
}

interface OllamaTool {
  type: 'function';
  function: OllamaToolFunction;
}

interface OllamaChatRequest {
  model: string;
  messages: OllamaMessage[];
  stream: boolean;
  think?: boolean;
  tools?: OllamaTool[];
  options?: {
    num_ctx?: number;
    temperature?: number;
    num_predict?: number;
  };
}

interface OllamaChatResponse {
  message: OllamaMessage;
  done: boolean;
}

// ---------------------------------------------------------------------------
// Gmail tool definitions
// ---------------------------------------------------------------------------

const GMAIL_TOOLS: OllamaTool[] = [
  {
    type: 'function',
    function: {
      name: 'gmail_list_emails',
      description: 'List recent emails from Gmail. Returns sender, subject, date, and snippet for each.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Gmail search query (e.g. "is:unread", "from:alice@example.com", "subject:invoice"). Defaults to recent primary inbox.',
          },
          max_results: {
            type: 'number',
            description: 'Maximum number of emails to return (1–20, default 10).',
          },
        },
        required: [],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'gmail_get_email',
      description: 'Get the full content of a specific email by its ID.',
      parameters: {
        type: 'object',
        properties: {
          email_id: {
            type: 'string',
            description: 'The Gmail message ID (from gmail_list_emails results).',
          },
        },
        required: ['email_id'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'gmail_send_email',
      description: 'Send a new email.',
      parameters: {
        type: 'object',
        properties: {
          to: { type: 'string', description: 'Recipient email address.' },
          subject: { type: 'string', description: 'Email subject line.' },
          body: { type: 'string', description: 'Plain text email body.' },
        },
        required: ['to', 'subject', 'body'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'gmail_reply_email',
      description: 'Reply to an existing email thread.',
      parameters: {
        type: 'object',
        properties: {
          email_id: {
            type: 'string',
            description: 'The Gmail message ID to reply to.',
          },
          body: { type: 'string', description: 'Plain text reply body.' },
        },
        required: ['email_id', 'body'],
      },
    },
  },
];

// ---------------------------------------------------------------------------
// Gmail client
// ---------------------------------------------------------------------------

interface GmailClient {
  gmail: gmail_v1.Gmail;
  userEmail: string;
}

let _gmailClient: GmailClient | null | undefined = undefined;

function getGmailClient(): GmailClient | null {
  if (_gmailClient !== undefined) return _gmailClient;

  const credDir = path.join(os.homedir(), '.gmail-mcp');
  const keysPath = path.join(credDir, 'gcp-oauth.keys.json');
  const tokensPath = path.join(credDir, 'credentials.json');

  if (!fs.existsSync(keysPath) || !fs.existsSync(tokensPath)) {
    _gmailClient = null;
    return null;
  }

  try {
    const keys = JSON.parse(fs.readFileSync(keysPath, 'utf-8'));
    const tokens = JSON.parse(fs.readFileSync(tokensPath, 'utf-8'));
    const clientConfig = keys.installed || keys.web || keys;
    const { client_id, client_secret, redirect_uris } = clientConfig;

    const oauth2Client = new google.auth.OAuth2(client_id, client_secret, redirect_uris?.[0]);
    oauth2Client.setCredentials(tokens);

    oauth2Client.on('tokens', (newTokens) => {
      try {
        const current = JSON.parse(fs.readFileSync(tokensPath, 'utf-8'));
        Object.assign(current, newTokens);
        fs.writeFileSync(tokensPath, JSON.stringify(current, null, 2));
      } catch { /* ignore */ }
    });

    _gmailClient = { gmail: google.gmail({ version: 'v1', auth: oauth2Client }), userEmail: '' };
    return _gmailClient;
  } catch (err) {
    logger.warn({ err }, 'Failed to initialize Gmail client for Ollama tools');
    _gmailClient = null;
    return null;
  }
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

function extractTextBody(payload: gmail_v1.Schema$MessagePart | undefined): string {
  if (!payload) return '';
  if (payload.mimeType === 'text/plain' && payload.body?.data) {
    return Buffer.from(payload.body.data, 'base64').toString('utf-8');
  }
  if (payload.parts) {
    for (const part of payload.parts) {
      if (part.mimeType === 'text/plain' && part.body?.data) {
        return Buffer.from(part.body.data, 'base64').toString('utf-8');
      }
    }
    for (const part of payload.parts) {
      const text = extractTextBody(part);
      if (text) return text;
    }
  }
  return '';
}

async function ensureUserEmail(client: GmailClient): Promise<void> {
  if (client.userEmail) return;
  const profile = await client.gmail.users.getProfile({ userId: 'me' });
  client.userEmail = profile.data.emailAddress || '';
}

async function toolListEmails(
  client: GmailClient,
  args: { query?: string; max_results?: number },
): Promise<unknown> {
  const maxResults = Math.min(args.max_results ?? 10, 20);
  const q = args.query || 'category:primary';

  const listRes = await client.gmail.users.messages.list({ userId: 'me', q, maxResults });
  const stubs = listRes.data.messages || [];
  if (stubs.length === 0) return { emails: [] };

  const emails = await Promise.all(
    stubs.map(async (stub) => {
      const msg = await client.gmail.users.messages.get({
        userId: 'me',
        id: stub.id!,
        format: 'metadata',
        metadataHeaders: ['From', 'Subject', 'Date'],
      });
      const headers = msg.data.payload?.headers || [];
      const getH = (n: string) => headers.find(h => h.name?.toLowerCase() === n.toLowerCase())?.value || '';
      return {
        id: stub.id,
        from: getH('From'),
        subject: getH('Subject'),
        date: getH('Date'),
        snippet: msg.data.snippet || '',
      };
    }),
  );

  return { emails };
}

async function toolGetEmail(
  client: GmailClient,
  args: { email_id: string },
): Promise<unknown> {
  const msg = await client.gmail.users.messages.get({
    userId: 'me',
    id: args.email_id,
    format: 'full',
  });

  const headers = msg.data.payload?.headers || [];
  const getH = (n: string) => headers.find(h => h.name?.toLowerCase() === n.toLowerCase())?.value || '';

  return {
    id: args.email_id,
    threadId: msg.data.threadId,
    from: getH('From'),
    to: getH('To'),
    subject: getH('Subject'),
    date: getH('Date'),
    body: extractTextBody(msg.data.payload).slice(0, 8000),
  };
}

function buildRawEmail(headers: string[], body: string): string {
  const raw = [...headers, '', body].join('\r\n');
  return Buffer.from(raw).toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

async function toolSendEmail(
  client: GmailClient,
  args: { to: string; subject: string; body: string },
): Promise<unknown> {
  await ensureUserEmail(client);
  const encoded = buildRawEmail([
    `To: ${args.to}`,
    `From: ${client.userEmail}`,
    `Subject: ${args.subject}`,
    'Content-Type: text/plain; charset=utf-8',
  ], args.body);

  await client.gmail.users.messages.send({ userId: 'me', requestBody: { raw: encoded } });
  return { success: true, to: args.to, subject: args.subject };
}

async function toolReplyEmail(
  client: GmailClient,
  args: { email_id: string; body: string },
): Promise<unknown> {
  await ensureUserEmail(client);

  const msg = await client.gmail.users.messages.get({
    userId: 'me',
    id: args.email_id,
    format: 'metadata',
    metadataHeaders: ['From', 'Subject', 'Message-ID'],
  });

  const headers = msg.data.payload?.headers || [];
  const getH = (n: string) => headers.find(h => h.name?.toLowerCase() === n.toLowerCase())?.value || '';
  const subject = getH('Subject');
  const replySubject = subject.startsWith('Re:') ? subject : `Re: ${subject}`;
  const messageId = getH('Message-ID');

  const encoded = buildRawEmail([
    `To: ${getH('From')}`,
    `From: ${client.userEmail}`,
    `Subject: ${replySubject}`,
    `In-Reply-To: ${messageId}`,
    `References: ${messageId}`,
    'Content-Type: text/plain; charset=utf-8',
  ], args.body);

  await client.gmail.users.messages.send({
    userId: 'me',
    requestBody: { raw: encoded, threadId: msg.data.threadId! },
  });

  return { success: true, replied_to: getH('From'), subject: replySubject };
}

async function executeTool(
  name: string,
  args: Record<string, unknown>,
  client: GmailClient,
): Promise<string> {
  try {
    let result: unknown;
    switch (name) {
      case 'gmail_list_emails':
        result = await toolListEmails(client, args as { query?: string; max_results?: number });
        break;
      case 'gmail_get_email':
        result = await toolGetEmail(client, args as { email_id: string });
        break;
      case 'gmail_send_email':
        result = await toolSendEmail(client, args as { to: string; subject: string; body: string });
        break;
      case 'gmail_reply_email':
        result = await toolReplyEmail(client, args as { email_id: string; body: string });
        break;
      default:
        result = { error: `Unknown tool: ${name}` };
    }
    return JSON.stringify(result);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logger.warn({ tool: name, err }, 'Tool execution error');
    return JSON.stringify({ error: msg });
  }
}

// ---------------------------------------------------------------------------
// Ollama API call
// ---------------------------------------------------------------------------

async function callOllama(
  model: string,
  messages: OllamaMessage[],
  tools: OllamaTool[],
): Promise<OllamaMessage> {
  const url = `${OLLAMA_BASE_URL}/api/chat`;
  const body: OllamaChatRequest = {
    model,
    messages,
    stream: false,
    think: true,
    tools: tools.length > 0 ? tools : undefined,
    options: { num_ctx: OLLAMA_NUM_CTX },
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
  return data.message;
}

// ---------------------------------------------------------------------------
// Tool-calling loop
// ---------------------------------------------------------------------------

const MAX_TOOL_ROUNDS = 10;

async function runTurn(
  model: string,
  messages: OllamaMessage[],
  tools: OllamaTool[],
  gmailClient: GmailClient | null,
): Promise<{ messages: OllamaMessage[]; text: string }> {
  const working = [...messages];

  for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
    const assistantMsg = await callOllama(model, working, tools);
    working.push(assistantMsg);

    if (!assistantMsg.tool_calls || assistantMsg.tool_calls.length === 0) {
      return { messages: working, text: assistantMsg.content || '' };
    }

    logger.info({ toolCount: assistantMsg.tool_calls.length, round }, 'Executing tool calls');

    for (const tc of assistantMsg.tool_calls) {
      const toolName = tc.function.name;
      logger.info({ tool: toolName }, 'Calling tool');

      const result = gmailClient
        ? await executeTool(toolName, tc.function.arguments, gmailClient)
        : JSON.stringify({ error: 'Gmail not configured' });

      working.push({ role: 'tool', content: result });
    }
  }

  logger.warn({ model }, 'Tool loop hit max rounds');
  return { messages: working, text: 'Reached the maximum number of tool calls without completing the task.' };
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

function historyPath(groupFolder: string): string {
  return path.join(DATA_DIR, 'sessions', groupFolder, 'ollama-history.json');
}

function loadHistory(groupFolder: string): OllamaMessage[] {
  const file = historyPath(groupFolder);
  try {
    if (fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, 'utf-8'));
  } catch { /* corrupted — start fresh */ }
  return [];
}

function saveHistory(groupFolder: string, messages: OllamaMessage[]): void {
  const file = historyPath(groupFolder);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(messages.slice(-100), null, 2));
}

function buildSystemPrompt(group: RegisteredGroup): string {
  return (
    `You are ${ASSISTANT_NAME}, a helpful personal assistant in a chat group called "${group.name}". ` +
    `Messages are provided in XML format with sender names and timestamps. ` +
    `Respond conversationally and helpfully. Keep responses concise unless detail is requested. ` +
    `Do not wrap your reply in XML tags.`
  );
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export async function runOllamaAgent(
  group: RegisteredGroup,
  prompt: string,
  model: string = OLLAMA_MODEL,
  onOutput?: (output: ContainerOutput) => Promise<void>,
): Promise<ContainerOutput> {
  const gmailClient = getGmailClient();
  const tools = gmailClient ? GMAIL_TOOLS : [];

  const history = loadHistory(group.folder);
  const messages: OllamaMessage[] = [
    { role: 'system', content: buildSystemPrompt(group) },
    ...history,
    { role: 'user', content: prompt },
  ];

  logger.info(
    { group: group.name, model, historyLength: history.length, tools: tools.map(t => t.function.name) },
    'Calling Ollama',
  );

  try {
    const { messages: updated, text } = await runTurn(model, messages, tools, gmailClient);

    // Persist history: everything after the system prompt (includes tool calls/results for context)
    saveHistory(group.folder, updated.slice(1));

    const output: ContainerOutput = { status: 'success', result: text };
    if (onOutput) {
      await onOutput(output);
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
