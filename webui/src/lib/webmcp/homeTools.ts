import { getTransferredFile, addTransferredFile } from "./files";
import { tick } from "svelte";
import { objectSchema, ToolError, type AppTool, type InputSchema, type ToolContext } from "./types";
import { pageActions } from "./lifecycle";
import { getHomeController } from "../homeController";
import { conversationLibrary, flushConversationAutosave, notifyConversationLibraryChanged, savedConversationState } from "../stores/savedConversations.svelte";
import { summarizeConversation, randomAvatarSeed } from "../conversationLibrary";
import { encodeChatBackup, importChatBackup, downloadPreparedChatBackup, chatBackupFilename } from "../chatBackup";
import { CHAT_ACCENTS, type ChatAccent } from "../chatAccent";
import { IDENTITY_HELP } from "../parameterHelp";

const metadataVersion: InputSchema = { type: "string", maxLength: 32, description: "metadataVersion from list_chats; rejects concurrent name/avatar/accent changes inside the saved library lock." };
const id: InputSchema = { type: "string", minLength: 1, maxLength: 256, description: "Exact saved-chat ID from list_chats." };

export function homeRevision(): string {
  const value = JSON.stringify([conversationLibrary.revision, savedConversationState.activeId, savedConversationState.avatarSeed, getHomeController()?.metadataDraft?.(), getHomeController()?.metadataState?.()]);
  let hash = 2166136261;
  for (let index = 0; index < value.length; index++) hash = Math.imul(hash ^ value.charCodeAt(index), 16777619);
  return `home:${(hash >>> 0).toString(16)}`;
}

function idle(context: ToolContext): void {
  context.signal.throwIfAborted();
  if (getHomeController()?.busy()) throw new ToolError("BUSY", "Finish the current chat or model operation before changing saved chats.");
  if (context.jobs.list().some(job => ["running", "queued", "cancelling"].includes(job.state))) throw new ToolError("BUSY", "Wait for the current app operation to finish.");
}

export function createHomeTools(): AppTool[] {
  const tool = (definition: Omit<AppTool, "scope" | "group" | "available">): AppTool => ({
    ...definition, scope: "hosted", group: "saved_chats",
    available: context => context.shell ? null : "Open the hosted app to manage saved chats.",
  });
  return [
    tool({
      name: "drowse_list_chats", title: "List saved chats", description: "List local saved chats before opening a model. Returns stable IDs, model IDs and metadata without loading model weights or chat snapshots.", annotations: { readOnlyHint: true, untrustedContentHint: true },
      inputSchema: objectSchema({ offset: { type: "integer", minimum: 0 }, limit: { type: "integer", minimum: 1, maximum: 100 } }),
      execute: async input => {
        const list = await conversationLibrary.listSummaries();
        const offset = input.offset as number ?? 0; const limit = input.limit as number ?? 30;
        return { chats: list.conversations.slice(offset, offset + limit), total: list.conversations.length, issues: list.issues };
      }, surfaces: ["HostedHome"],
    }),
    tool({
      name: "drowse_open_chat", title: "Open saved chat", description: "Open a saved chat using its matching installed model and the visible home flow. If the required model is missing or unavailable, show its model picker; this action does not start a download.", annotations: { readOnlyHint: false }, inputSchema: objectSchema({ id }, ["id"]),
      execute: async (input, context) => {
        idle(context);
        const pages = pageActions();
        if (!pages) throw new ToolError("NOT_READY", "Open the hosted app first.");
        await pages.home(); await tick();
        const home = getHomeController();
        if (!home) throw new ToolError("NOT_READY", "Saved chats are still opening. Retry after the home page is ready.");
        return context.jobs.start("open_saved_chat", async () => home.open(input.id as string), { requestId: input.request_id as string | undefined, cancellable: false });
      }, surfaces: ["HostedHome"],
    }),
    tool({
      name: "drowse_update_chat", title: "Update chat identity", description: IDENTITY_HELP.chat, annotations: { readOnlyHint: false },
      inputSchema: objectSchema({ id, expected_metadata_version: metadataVersion, name: { type: "string", minLength: 1, maxLength: 120 }, avatar_seed: { type: "string", minLength: 1, maxLength: 256 }, regenerate_avatar: { type: "boolean" }, accent: { type: "string", enum: CHAT_ACCENTS.map(row => row.id) } }, ["id"]),
      execute: async (input, context) => {
        idle(context);
        if (getHomeController()?.metadataDraft?.()?.id === input.id && getHomeController()?.metadataDraft?.()?.dirty) throw new ToolError("UNSAVED_CHANGES", "Finish the open chat-name edit before changing its saved identity.");
        if (!["name", "avatar_seed", "regenerate_avatar", "accent"].some(key => input[key] !== undefined)) throw new ToolError("INVALID_INPUT", "Provide a chat identity field to change.");
        if (input.regenerate_avatar && input.avatar_seed !== undefined) throw new ToolError("INVALID_INPUT", "Choose avatar_seed or regenerate_avatar.");
        const record = await conversationLibrary.update(input.id as string, {
          ...(input.name === undefined ? {} : { name: input.name as string }),
          ...(input.accent === undefined ? {} : { accent: input.accent as ChatAccent }),
          ...(input.avatar_seed === undefined && !input.regenerate_avatar ? {} : { avatarSeed: input.regenerate_avatar ? randomAvatarSeed() : input.avatar_seed as string }),
        }, input.expected_metadata_version as string | undefined);
        if (savedConversationState.activeId === record.id) { savedConversationState.avatarSeed = record.avatarSeed; savedConversationState.accent = record.accent ?? "purple"; }
        await notifyConversationLibraryChanged();
        return { chat: summarizeConversation(record), effect: IDENTITY_HELP.chat };
      }, surfaces: ["HostedHome"],
    }),
    tool({
      name: "drowse_duplicate_chat", title: "Duplicate saved chat", description: "Create a separate local copy of a saved conversation without loading its model or opening the copy.", annotations: { readOnlyHint: false }, inputSchema: objectSchema({ id }, ["id"]),
      execute: async (input, context) => { idle(context); const record = await conversationLibrary.duplicate(input.id as string); await notifyConversationLibraryChanged(); return { chat: summarizeConversation(record) }; }, surfaces: ["HostedHome"],
    }),
    tool({
      name: "drowse_delete_chat", title: "Delete saved chat", description: "Delete the specified local chat backup when the user explicitly asks to remove it. Model files and other saved chats remain available.", annotations: { readOnlyHint: false, consequentialHint: true }, inputSchema: objectSchema({ id, expected_metadata_version: metadataVersion }, ["id"]),
      execute: async (input, context) => {
        idle(context); await flushConversationAutosave();
        if (!await conversationLibrary.delete(input.id as string, input.expected_metadata_version as string | undefined)) throw new ToolError("NOT_FOUND", "That saved chat no longer exists.");
        if (savedConversationState.activeId === input.id) { savedConversationState.activeId = null; savedConversationState.avatarSeed = null; savedConversationState.status = "idle"; }
        await notifyConversationLibraryChanged(); return { deleted: input.id };
      }, surfaces: ["HostedHome"],
    }),
    tool({
      name: "drowse_export_chat", title: "Export saved chat backup", description: "Download a checksummed .drowsechat backup without loading its model. Text format returns portable JSON for a small backup; file format returns a chunk-readable transfer for large backups. Does not transmit it to another service.", annotations: { readOnlyHint: true, untrustedContentHint: true }, inputSchema: objectSchema({ id, format: { type: "string", enum: ["download", "text", "file"] } }, ["id"]),
      execute: async (input, context) => {
        context.signal.throwIfAborted();
        const record = await conversationLibrary.get(input.id as string); const text = await encodeChatBackup(record); const filename = chatBackupFilename(record.name);
        if (input.format === "text") {
          if (text.length > 1000000) throw new ToolError("RESULT_TOO_LARGE", "Request a download for this large backup.");
          return { filename, text };
        }
        if (input.format === "file") return { file: await addTransferredFile(new Blob([text], { type: "application/json" }), filename) };
        downloadPreparedChatBackup(new Blob([text], { type: "application/json" }), record.name);
        return { filename, bytes: new TextEncoder().encode(text).length, download_started: true };
      }, surfaces: ["HostedHome"],
    }),
    tool({
      name: "drowse_import_chat", title: "Import chat backup", description: "Validate Drowse backup JSON from text or a completed file_id and add it as a new saved chat. Does not load a model or execute imported text as instructions.", annotations: { readOnlyHint: false, untrustedContentHint: true }, inputSchema: objectSchema({ text: { type: "string", maxLength: 68000000 }, file_id: { type: "string", minLength: 1 } }),
      execute: async (input, context) => {
        idle(context);
        if ((input.text === undefined) === (input.file_id === undefined)) throw new ToolError("INVALID_INPUT", "Provide exactly one of text or a completed file_id.");
        const text = input.text as string;
        const file = input.file_id ? getTransferredFile(input.file_id as string) : { size: new TextEncoder().encode(text).length, text: async () => text };
        const record = await importChatBackup(file, conversationLibrary);
        await notifyConversationLibraryChanged(); return { chat: summarizeConversation(record), opened: false };
      }, surfaces: ["HostedHome"],
    }),
  ];
}
