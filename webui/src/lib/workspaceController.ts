import type { InputSchema } from "./webmcp/types";
import type { ChatRole } from "./types";

export interface WorkspaceNavigation {
  view?: "conversation" | "branches" | "controls" | "tools";
  section?: "response" | "model" | "chat";
  leftSidebar?: boolean;
  headersVisible?: boolean;
}

export interface ComposerDraft {
  text: string;
  authoredRole: ChatRole;
  generatedRole: ChatRole | null;
  authoredThinking: string;
}

export interface RawBufferDraft {
  text: string;
  dirty: boolean;
  busy: boolean;
  selection: { start: number; end: number } | null;
}

interface WorkspaceController {
  read(): Required<WorkspaceNavigation>;
  navigate(change: WorkspaceNavigation): void | Promise<void>;
}

interface ComposerController {
  read(): ComposerDraft;
  update(change: Partial<ComposerDraft>): void | Promise<void>;
}

interface RawBufferController {
  read(): RawBufferDraft;
  update(change: { text?: string; selection?: { start: number; end: number } }): void | Promise<void>;
  save(): Promise<void>;
  revert(): void;
  prepareSelection(recomplete: boolean): Promise<{ parent_node_id: string }>;
}

interface SettingsFormController {
  read(): { busy: boolean; dirty: boolean; values: Record<string, unknown> };
  sync(): void;
}

export interface InterfaceController {
  schema: InputSchema;
  read(): Record<string, unknown>;
  update(change: Record<string, unknown>): void | Promise<void>;
}

const interfaces = new Map<string, InterfaceController>();
export function registerInterfaceController(id: string, controller: InterfaceController): () => void {
  interfaces.set(id, controller);
  return () => { if (interfaces.get(id) === controller) interfaces.delete(id); };
}
export function getInterfaceController(id: string): InterfaceController | null { return interfaces.get(id) ?? null; }
export function readInterfaceStates(): Record<string, Record<string, unknown>> {
  return Object.fromEntries([...interfaces].map(([id, controller]) => [id, { ...controller.read(), schema: controller.schema }]));
}
export function readRegisteredForms(): Record<string, Record<string, unknown>> {
  return Object.fromEntries([...interfaces].filter(([id]) => id.startsWith("manifold_") || id === "template_lab" || id === "chat_identity").map(([id, controller]) => [id, { ...controller.read(), schema: controller.schema }]));
}

function controllerSlot<T>() {
  let controller: T | null = null;
  return {
    get: () => controller,
    register(value: T) {
      controller = value;
      return () => { if (controller === value) controller = null; };
    },
  };
}

const workspace = controllerSlot<WorkspaceController>();
const composer = controllerSlot<ComposerController>();
const rawBuffer = controllerSlot<RawBufferController>();
const cast = controllerSlot<SettingsFormController>();
const systemPrompt = controllerSlot<SettingsFormController>();

export const registerWorkspaceController = workspace.register;
export const getWorkspaceController = workspace.get;
export const registerComposerController = composer.register;
export const getComposerController = composer.get;
export const registerRawBufferController = rawBuffer.register;
export const getRawBufferController = rawBuffer.get;
export const registerCastController = cast.register;
export const getCastController = cast.get;
export const registerSystemPromptController = systemPrompt.register;
export const getSystemPromptController = systemPrompt.get;
