import type { HostedController, RuntimeCapabilities, RuntimeClient } from "../runtime/contracts";
import type { HostedShellController } from "../../hosted/ui/types";
import type { JobRegistry } from "./jobs";
import type { SessionInfo } from "../types";

export interface InputSchema {
  type?: string | string[];
  description?: string;
  properties?: Record<string, InputSchema>;
  required?: string[];
  additionalProperties?: boolean | InputSchema;
  items?: InputSchema;
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
  minItems?: number;
  maxItems?: number;
  minLength?: number;
  maxLength?: number;
  anyOf?: InputSchema[];
}

export type ToolScope = "public" | "hosted" | "workspace" | "page";

export interface ToolContext {
  signal: AbortSignal;
  jobs: JobRegistry;
  runtime: RuntimeClient | null;
  hosted: HostedController | null;
  shell: HostedShellController | null;
  capabilities: RuntimeCapabilities | null;
  assertWorkspaceIdle?: () => void;
  session?: SessionInfo | null;
}

export interface AppTool {
  name: string;
  title: string;
  description: string;
  inputSchema: InputSchema;
  annotations: {
    readOnlyHint: boolean;
    untrustedContentHint?: boolean;
    consequentialHint?: boolean;
  };
  scope: ToolScope;
  group: string;
  surfaces?: string[];
  services?: string[];
  available?: (context: ToolContext) => string | null;
  execute: (input: Record<string, unknown>, context: ToolContext) => unknown | Promise<unknown>;
}

export class ToolError extends Error {
  constructor(public code: string, message: string, public details?: unknown) {
    super(message);
    this.name = "ToolError";
  }
}

export const objectSchema = (
  properties: Record<string, InputSchema> = {},
  required: string[] = [],
): InputSchema => ({ type: "object", properties, required, additionalProperties: false });
