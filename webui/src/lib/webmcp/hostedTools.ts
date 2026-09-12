import { objectSchema, ToolError, type AppTool, type ToolContext } from "./types";
import { pageActions, selectVisibleModel } from "./lifecycle";

const modelId = { type: "string", minLength: 1, description: "Exact model variant ID from drowse_list_models." };
const packId = { type: "string", minLength: 1, description: "Exact compatible pack ID from the selected model." };

function shell(context: ToolContext) {
  if (!context.shell) throw new ToolError("unavailable", "Open the hosted app's model picker first.");
  return context.shell;
}

function model(context: ToolContext, id: string) {
  const selected = shell(context).current().models.find(candidate => candidate.id === id);
  if (!selected) throw new ToolError("not_found", "Model variant is not in the verified catalog. List models first.");
  return selected;
}

function idle(context: ToolContext): void {
  context.assertWorkspaceIdle?.();
  if (context.jobs.list().some(job => ["queued", "running", "cancelling"].includes(job.state))) {
    throw new ToolError("busy", "Wait for the current application job or cancel it before changing model setup.");
  }
  const state = context.hosted?.snapshot;
  if (state && [state.download, state.generation, state.fitting].some(operation => ["running", "cancelling"].includes(operation.phase))) {
    throw new ToolError("busy", "Wait for current generation, download, or fitting to finish, or cancel its job first.");
  }
  if (state?.lifecycle === "loading") throw new ToolError("busy", "The model is still loading.");
}

function confirmAction(message: string): void {
  if (!window.confirm(message)) throw new ToolError("cancelled", "The user cancelled this action.");
}

export function createHostedTools(): AppTool[] {
  const tool = (definition: Omit<AppTool, "scope" | "group" | "available"> & { available?: AppTool["available"] }): AppTool => ({
    scope: "hosted", group: "models", available: context => context.shell ? null : "Open the hosted app first.", ...definition,
  });
  return [
    tool({
      name: "drowse_list_models", title: "List eligible models", annotations: { readOnlyHint: true, untrustedContentHint: true },
      description: "List verified model variants, instruction/base type, eligibility, storage/download sizes, selected packs and setup status. Blocked models cannot be loaded.",
      inputSchema: objectSchema({ query: { type: "string" }, installed: { type: "boolean" }, limit: { type: "integer", minimum: 1, maximum: 50 }, offset: { type: "integer", minimum: 0 } }),
      execute: (input, context) => {
        const snapshot = shell(context).current();
        const query = String(input.query ?? "").toLowerCase();
        const rows = snapshot.models.filter(row => `${row.id} ${row.name} ${row.language}`.toLowerCase().includes(query) && (input.installed === undefined || row.installed === input.installed));
        const offset = Number(input.offset ?? 0), limit = Number(input.limit ?? 10);
        return { phase: snapshot.phase, models: rows.slice(offset, offset + limit), total: rows.length, next_offset: offset + limit < rows.length ? offset + limit : null, storage: snapshot.storage, download: snapshot.download, runtime: snapshot.runtime };
      },
    }),
    tool({
      name: "drowse_check_device", title: "Check this device", annotations: { readOnlyHint: false },
      description: "Run graphics/storage checks and refresh verified model recommendations. This performs a short device check, not model inference.", inputSchema: objectSchema(),
      execute: (input, context) => {
        idle(context);
        return context.jobs.start("device_check", async () => {
          await shell(context).check();
          const snapshot = shell(context).current();
          if (snapshot.phase === "failed") throw new ToolError("device_check_failed", snapshot.detail);
          return snapshot;
        }, { requestId: input.request_id as string | undefined, cancellable: false });
      },
    }),
    tool({
      name: "drowse_select_model", title: "Select model and packs", annotations: { readOnlyHint: false },
      description: "Select a catalog model and optional first-run packs in the visible picker. This does not download or load the model. Required packs stay selected.",
      inputSchema: objectSchema({ model_id: modelId, packs: { type: "array", maxItems: 1, description: "Change one optional pack at a time through the same validated picker action.", items: objectSchema({ pack_id: packId, selected: { type: "boolean" } }, ["pack_id", "selected"]) } }, ["model_id"]),
      execute: async (input, context) => {
        idle(context);
        const selected = model(context, input.model_id as string);
        const packs = (input.packs ?? []) as Array<{ pack_id: string; selected: boolean }>;
        for (const item of packs) {
          const pack = selected.firstRunPacks.find(candidate => candidate.id === item.pack_id);
          if (!pack) throw new ToolError("not_found", "This pack is not a first-run option for the selected model.");
          if (pack.requiredForSetup && !item.selected) throw new ToolError("invalid_input", "Required setup packs must remain selected.");
        }
        if (!selectVisibleModel(selected.id)) {
          await pageActions()?.models();
          await (await import("svelte")).tick();
          if (!selectVisibleModel(selected.id)) throw new ToolError("unavailable", "Open the model picker and retry selection.");
        }
        for (const item of packs) shell(context).setOptionalPackSelected(selected.id, item.pack_id, item.selected);
        return { selected_model_id: selected.id, model: model(context, selected.id) };
      },
    }),
    tool({
      name: "drowse_download_model", title: "Download or resume model", annotations: { readOnlyHint: false },
      description: "Download/resume a verified model with its selected setup packs. Returns a job with real progress. Conditional device eligibility uses the app's confirmation; blocked models remain blocked.",
      inputSchema: objectSchema({ model_id: modelId }, ["model_id"]),
      execute: (input, context) => {
        idle(context);
        const selected = model(context, input.model_id as string);
        if (selected.fit === "blocked") throw new ToolError("unavailable", selected.reason);
        if (selected.fit === "uncertain") confirmAction(`${selected.name}: ${selected.reason}\nDownload and try this model on this device?`);
        selectVisibleModel(selected.id);
        return context.jobs.start("model_download", async job => {
          const unsubscribe = shell(context).subscribe(snapshot => job.progress(snapshot.download));
          try {
            await shell(context).download(selected.id, { explicitUnsafeOverride: selected.fit === "uncertain" });
            const snapshot = shell(context).current();
            if (snapshot.download.phase === "paused") throw new DOMException("The download was paused; call download again to resume verified bytes.", "AbortError");
            if (snapshot.download.phase !== "installed") throw new ToolError("download_incomplete", snapshot.download.reason);
            return snapshot;
          }
          finally { unsubscribe(); }
        }, { requestId: input.request_id as string | undefined, cancel: () => shell(context).cancelDownload() });
      },
    }),
    tool({
      name: "drowse_load_model", title: "Open installed model", annotations: { readOnlyHint: false },
      description: "Open an installed model through the normal setup/workbench flow. Loading does not train it. Optional reset replaces the working session after confirmation; saved chats remain separate.",
      inputSchema: objectSchema({ model_id: modelId, reset_session: { type: "boolean" } }, ["model_id"]),
      execute: (input, context) => {
        idle(context);
        const selected = model(context, input.model_id as string);
        if (selected.fit === "blocked") throw new ToolError("unavailable", selected.reason);
        if (!selected.setupComplete) throw new ToolError("prerequisite", "Download the model and its required setup packs first.");
        if (input.reset_session) confirmAction(`Reset the working session for ${selected.name}? Saved chats are kept separately.`);
        if (selected.fit === "uncertain" || selected.requiresOomRetry) confirmAction(`${selected.name}: ${selected.reason}\nTry loading this model now?`);
        selectVisibleModel(selected.id);
        return context.jobs.start("model_load", async () => { await shell(context).open(selected.id, { resetSession: input.reset_session === true }); return shell(context).current(); }, { requestId: input.request_id as string | undefined, cancellable: false });
      },
    }),
    tool({
      name: "drowse_unload_model", title: "Unload model", annotations: { readOnlyHint: false },
      description: "Save the current hosted workspace and unload model memory. Downloaded files and saved chats remain available.", inputSchema: objectSchema(),
      execute: async (_input, context) => { idle(context); await shell(context).prepareForReload(); await shell(context).unload(); return { state: "unloaded" }; },
    }),
    tool({
      name: "drowse_download_pack", title: "Download analysis pack", annotations: { readOnlyHint: false },
      description: "Download a compatible J/R-lens or SAE pack. Installation can require reloading the model before activation; it does not fit or train an instrument.",
      inputSchema: objectSchema({ model_id: modelId, pack_id: packId }, ["model_id", "pack_id"]),
      execute: (input, context) => {
        idle(context);
        model(context, input.model_id as string);
        const controller = context.hosted ?? shell(context).hostedController();
        return context.jobs.start("pack_download", async job => {
          const result = await controller.downloadPack(input.model_id as string, input.pack_id as string, job.progress);
          if (result.cancelled) throw new DOMException("The pack download was cancelled", "AbortError");
          if (!result.installed) throw new ToolError("download_incomplete", "The analysis pack did not finish installation.");
          return { ...result, reload_required: controller.snapshot.lifecycle === "ready" && controller.snapshot.modelVariantId === input.model_id };
        }, { requestId: input.request_id as string | undefined, cancel: () => controller.cancelDownload() });
      },
    }),
    tool({
      name: "drowse_read_storage", title: "Read local model storage", annotations: { readOnlyHint: true },
      description: "Read storage estimates, persistence protection, installed models and analysis packs. Estimates can change as the browser manages storage.", inputSchema: objectSchema(),
      execute: async (_input, context) => {
        const controller = context.hosted ?? shell(context).hostedController();
        return { storage: await controller.refreshStorage(), models: controller.snapshot.installedModelVariantIds, packs: controller.snapshot.installedPackIds };
      },
    }),
    tool({
      name: "drowse_protect_storage", title: "Request storage protection", annotations: { readOnlyHint: false },
      description: "Ask the browser to protect local model/chat data from automatic eviction. Permission may require a browser interaction and can be denied.", inputSchema: objectSchema(),
      execute: async (_input, context) => ({ persisted: await shell(context).retryPersistence() }),
    }),
    tool({
      name: "drowse_delete_local_data", title: "Remove local model data", annotations: { readOnlyHint: false, consequentialHint: true },
      description: "Remove a selected model, analysis pack, or all hosted runtime data after the app's confirmation. Read storage first. Saved-chat library deletion is a separate action.",
      inputSchema: objectSchema({ kind: { type: "string", enum: ["model", "pack", "all"] }, id: { type: "string", minLength: 1 } }, ["kind"]),
      execute: async (input, context) => {
        idle(context);
        const controller = context.hosted ?? shell(context).hostedController();
        if (input.kind !== "all" && !input.id) throw new ToolError("invalid_input", "Specify the exact model or pack ID to remove.");
        if (input.kind === "model") model(context, input.id as string);
        if (input.kind === "pack" && !controller.snapshot.installedPackIds.includes(input.id as string)) throw new ToolError("not_found", "The analysis pack is not installed.");
        confirmAction(input.kind === "all" ? "Remove all hosted models, fitted runtime artifacts, and the working session from this device? Export needed work first." : `Remove ${input.kind} ${input.id} from this device?`);
        if (input.kind === "model") await shell(context).deleteModel(input.id as string);
        else if (input.kind === "pack") await controller.deletePack(input.id as string);
        else await controller.clearAll();
        return { state: "removed", kind: input.kind, id: input.id ?? null };
      },
    }),
    tool({
      name: "drowse_runtime_recovery", title: "Manage runtime recovery", annotations: { readOnlyHint: false, consequentialHint: true },
      description: "Retry workbench recovery or respond to an existing request to transfer model ownership to another tab. Allowing takeover interrupts this tab's model work.",
      inputSchema: objectSchema({ action: { type: "string", enum: ["retry", "allow_takeover", "deny_takeover"] } }, ["action"]),
      execute: async (input, context) => {
        const actions = pageActions();
        if (!actions) throw new ToolError("unavailable", "Open the hosted app first.");
        if (input.action === "retry") await actions.retry();
        else {
          if (shell(context).current().takeover?.phase !== "requested") throw new ToolError("unavailable", "There is no pending takeover request.");
          if (input.action === "allow_takeover") { confirmAction("Allow the other tab to take over this model, stopping this tab's model work?"); await actions.allowTakeover(); }
          else actions.denyTakeover();
        }
        return { action: input.action, page: actions.state(), takeover: shell(context).current().takeover ?? null };
      },
    }),
  ];
}
