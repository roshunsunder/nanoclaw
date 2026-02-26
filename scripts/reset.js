#!/usr/bin/env node
/**
 * Reset nanoclaw conversation state.
 *
 * Usage:
 *   npm run reset              # clear messages DB + Ollama history for main
 *   npm run reset:history      # clear only Ollama history (keep message DB)
 *   npm run reset -- --group <folder>  # target a different group folder
 */
import { execSync } from 'child_process';
import Database from 'better-sqlite3';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

const args = process.argv.slice(2);
const historyOnly = args.includes('--history-only');
const groupIdx = args.indexOf('--group');
const groupFolder = groupIdx !== -1 ? args[groupIdx + 1] : 'main';

const dbPath = path.join(root, 'store', 'messages.db');
const historyPath = path.join(root, 'data', 'sessions', groupFolder, 'ollama-history.json');

// Resolve the WhatsApp JID for this group from the DB
const db = new Database(dbPath);

function getJidForGroup(folder) {
  // registered_groups stores group config keyed by JID
  const rows = db.prepare("SELECT key, value FROM router_state WHERE key = 'registered_groups'").all();
  if (rows.length === 0) return null;
  try {
    const groups = JSON.parse(rows[0].value);
    return Object.entries(groups).find(([, g]) => g.folder === folder)?.[0] ?? null;
  } catch {
    return null;
  }
}

console.log(`Resetting group: ${groupFolder}${historyOnly ? ' (history only)' : ''}`);

// Stop service
try {
  execSync('systemctl --user stop nanoclaw', { stdio: 'inherit' });
} catch {
  // Not running — fine
}

// Clear Ollama history
if (fs.existsSync(historyPath)) {
  fs.unlinkSync(historyPath);
  console.log(`Deleted ${historyPath}`);
} else {
  console.log('No Ollama history file found — already clean.');
}

if (!historyOnly) {
  const jid = getJidForGroup(groupFolder);
  if (!jid) {
    console.warn(`Could not find JID for group "${groupFolder}" — skipping message DB clear.`);
  } else {
    const { changes } = db.prepare('DELETE FROM messages WHERE chat_jid = ?').run(jid);
    console.log(`Deleted ${changes} messages for ${jid}`);

    // Reset agent cursor for this group
    const row = db.prepare("SELECT value FROM router_state WHERE key = 'last_agent_timestamp'").get();
    if (row) {
      const ts = JSON.parse(row.value);
      delete ts[jid];
      db.prepare("UPDATE router_state SET value = ? WHERE key = 'last_agent_timestamp'").run(JSON.stringify(ts));
      console.log('Reset agent cursor.');
    }
  }
}

db.close();

// Restart service
execSync('systemctl --user start nanoclaw', { stdio: 'inherit' });
console.log('Done. nanoclaw restarted fresh.');
