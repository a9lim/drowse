import { tick } from "svelte";
import { objectSchema, ToolError, type AppTool, type InputSchema } from "./types";
import { validateInput } from "./validation";
import { getInterfaceController, readInterfaceStates } from "../workspaceController";
import { loomTree } from "../stores/loom.svelte";
import { loomUiState, filterState, applyTreeFilter, pinnedComparison, pinNodeForComparison, nodeSelection } from "../stores/loomUi.svelte";
import { lensState, saeState, setLensWorkspaceSortMode, setSaeSortMode } from "../stores/instruments.svelte";
import { cancelPendingAction, pendingActions } from "../stores/pending.svelte";
import { loomViewSchema, tokenInspectorSchema } from "../interfaceSchemas";

const formIds = ["manifold_builder", "manifold_discover", "manifold_authored", "manifold_templated", "manifold_merge", "template_lab", "chat_identity"];
const enumSchema = (values: string[]): InputSchema => ({ type: "string", enum: values });
const fields: InputSchema = { type: "object", additionalProperties: true, description: "Fields from this interface's returned schema. Only declared fields are accepted; omitted fields retain the human's draft." };

async function updateInterface(id: string, change: Record<string, unknown>) {
  const controller = getInterfaceController(id);
  if (!controller) throw new ToolError("NOT_READY", `Open the ${id} interface before changing its visible state.`);
  if (!Object.keys(change).length) throw new ToolError("INVALID_INPUT", "Provide at least one interface field.");
  validateInput(controller.schema, change);
  await controller.update(change);
  await tick();
  return getInterfaceController(id)?.read() ?? { closed: true };
}

export function createWorkspaceUiTools(): AppTool[] {
  const tool = (definition: Omit<AppTool, "scope" | "group"> & { group?: string }): AppTool => ({ scope: "workspace", group: "interface", ...definition });
  return [
    tool({ name: "drowse_set_loom_view", title: "Set visible Loom controls", description: "Apply the Loom sidebar's actual text search or advanced filter, sibling sort, selected nodes, pinned comparison, collapsed branches and map camera. Search results update the visible map. Read workspace for the current state. Camera controls require the visible map.", annotations: { readOnlyHint: false }, surfaces: ["LoomSidebar"],
      inputSchema: objectSchema({ ...loomViewSchema.properties, filter: objectSchema({ expression: { type: "string", maxLength: 20000 }, mode: enumSchema(["text", "advanced"]) }, ["expression"]), sort: { ...enumSchema(["default", "surprise", "confidence"]), description: "Sibling ordering by recorded model likelihood: confidence means token probability, not factual confidence or correctness." }, pin_node_id: { anyOf: [{ type: "string" }, { type: "null" }] }, selected_node_ids: { type: "array", items: { type: "string" }, maxItems: 1000 }, comparison_visible: { type: "boolean" } }),
      execute: async (input, context) => {
        context.signal.throwIfAborted();
        for (const id of [...(input.selected_node_ids as string[] ?? []), ...(typeof input.pin_node_id === "string" ? [input.pin_node_id] : [])]) if (!loomTree.nodes.has(id)) throw new ToolError("NOT_FOUND", `Tree node ${id} does not exist.`);
        const local = Object.fromEntries(Object.entries(input).filter(([key]) => key in (loomViewSchema.properties ?? {})));
        if (Object.keys(local).length) {
          const controller = getInterfaceController("loom");
          if (!controller) throw new ToolError("NOT_READY", "Open Loom before changing map controls.");
          validateInput(controller.schema, local);
        }
        if (input.comparison_visible !== undefined && !getInterfaceController("chat")) throw new ToolError("NOT_READY", "Open Conversation to change comparison visibility.");
        if (input.filter) { const filter = input.filter as { expression: string; mode?: "text" | "advanced" }; await applyTreeFilter(filter.expression, filter.mode ?? filterState.mode); if (filterState.error) throw new ToolError("INVALID_FILTER", filterState.error); }
        if (input.sort !== undefined) loomUiState.siblingSort = input.sort as typeof loomUiState.siblingSort;
        if (input.selected_node_ids !== undefined) nodeSelection.ids = [...new Set(input.selected_node_ids as string[])];
        if (input.pin_node_id !== undefined) pinNodeForComparison(input.pin_node_id as string | null);
        if (input.comparison_visible !== undefined) await updateInterface("chat", { comparison_visible: input.comparison_visible });
        await tick();
        if (Object.keys(local).length) await updateInterface("loom", local);
        return { filter: { mode: filterState.mode, expression: filterState.expr, matching_node_ids: filterState.matchingIds ? [...filterState.matchingIds] : null }, sort: loomUiState.siblingSort, pinned_node_id: pinnedComparison.nodeId, selected_node_ids: nodeSelection.ids, map: getInterfaceController("loom")?.read() ?? null };
      },
    }),
    tool({ name: "drowse_set_token_inspector", title: "Set token inspector", description: "Walk the visible token cursor, select primary or shadow branch, change analysis tab, apply the recorded steering recipe or inspect an unsteered replay, and dock, undock or hide details. Does not fork or generate text. Shadow selection requires a comparison on that exact turn.", annotations: { readOnlyHint: false }, inputSchema: tokenInspectorSchema, surfaces: ["token_drilldown"], execute: (input, context) => { context.signal.throwIfAborted(); return updateInterface("token_inspector", Object.fromEntries(Object.entries(input).filter(([key]) => key in (tokenInspectorSchema.properties ?? {})))); } }),
    tool({ name: "drowse_set_analysis_view", title: "Set analysis display", description: "Change display selections using the same controls as the visible interface. Lens sorts: strength/name/depth; SAE sorts: strength/name. Other targets return their accepted schema in read_workspace.interfaces and require the matching drawer open. Changes affect presentation; they do not modify model weights or steering.", annotations: { readOnlyHint: false }, inputSchema: objectSchema({ target: enumSchema(["lens", "sae", "probe_geometry", "node_compare", "profile_compare"]), fields }, ["target", "fields"]), surfaces: ["JLensPanel", "SaePanel", "probe_inspector", "node_compare", "compare"],
      execute: async (input, context) => {
        context.signal.throwIfAborted(); const target = input.target as string; const change = input.fields as Record<string, unknown>;
        if (target === "lens" || target === "sae") {
          validateInput(objectSchema({ sort: enumSchema(target === "lens" ? ["strength", "name", "depth"] : ["strength", "name"]) }, ["sort"]), change);
          if (target === "lens") setLensWorkspaceSortMode(change.sort as typeof lensState.workspaceSortMode); else setSaeSortMode(change.sort as typeof saeState.sortMode);
          await tick(); return { target, sort: target === "lens" ? lensState.workspaceSortMode : saeState.sortMode };
        }
        return updateInterface(target, change);
      },
    }),
    tool({ name: "drowse_cancel_pending_action", title: "Cancel queued action", description: "Cancel a specific human or agent action still waiting in the visible pending queue. Uses its normal cancellation callback and restores queued composer input where the UI supports it. Use drowse_cancel_job for an agent-owned running job; the Stop button controls human-owned running inference.", annotations: { readOnlyHint: false }, inputSchema: objectSchema({ id: { type: "string", minLength: 1 } }, ["id"]), surfaces: ["Chat", "RawBuffer"], execute: async (input, context) => { context.signal.throwIfAborted(); if (!pendingActions.queue.some(row => row.id === input.id)) throw new ToolError("NOT_FOUND", "This queued action already ran or was removed."); cancelPendingAction(input.id as string); await tick(); return { cancelled: input.id, pending: pendingActions.queue.map(row => ({ id: row.id, label: row.label })) }; } }),
    tool({ group: "authoring", name: "drowse_read_form", title: "Read unsaved authoring forms", description: "Read current mounted authoring and template form drafts, exact editable schema, validation and busy state. Text is untrusted user content. Open a form with navigate first; hidden visited builder tabs retain their drafts. Does not create or fit an artifact.", annotations: { readOnlyHint: true, untrustedContentHint: true }, inputSchema: objectSchema({ form: enumSchema(formIds) }), surfaces: ["manifold_builder", "manifold_merge", "template_lab"], execute: input => {
      if (input.form && !getInterfaceController(input.form as string)) throw new ToolError("NOT_READY", "Open this form (and its builder tab) before reading its draft.");
      return { forms: Object.fromEntries(formIds.filter(id => (!input.form || input.form === id) && getInterfaceController(id)).map(id => [id, { ...getInterfaceController(id)!.read(), schema: getInterfaceController(id)!.schema }])) };
    } }),
    tool({ group: "authoring", name: "drowse_edit_form", title: "Edit unsaved authoring form", description: "Patch an actual authoring/template draft using fields from read_form.schema. Omitted fields retain current human edits. Returns normal UI validation; incomplete drafts are allowed. Does not submit generation, fitting, merging, scoring or artifact creation. Use expected_revision from get_state to protect against concurrent edits.", annotations: { readOnlyHint: false, untrustedContentHint: true }, inputSchema: objectSchema({ form: enumSchema(formIds), fields }, ["form", "fields"]), surfaces: ["manifold_builder", "manifold_merge", "template_lab"], execute: async (input, context) => { context.assertWorkspaceIdle?.(); context.signal.throwIfAborted(); return updateInterface(input.form as string, input.fields as Record<string, unknown>); } }),
  ];
}

export function readWorkspaceInterfaces() {
  return { ...readInterfaceStates(), lens: { values: { sort: lensState.workspaceSortMode } }, sae: { values: { sort: saeState.sortMode } }, loom_filter: { expression: filterState.expr, mode: filterState.mode, sort: loomUiState.siblingSort, selected_node_ids: nodeSelection.ids, pin_node_id: pinnedComparison.nodeId } };
}
