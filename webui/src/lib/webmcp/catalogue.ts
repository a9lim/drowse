import { RUNTIME_SERVICE_METHODS } from "../runtime/contracts";
import type { DrawerName } from "../types";

export type CoverageEntry = {
  kind: "tool" | "unavailable" | "internal" | "interaction";
  tools: string[];
  reason?: string;
};

const exposed = (...tools: string[]): CoverageEntry => ({ kind: "tool", tools });
const internal = (reason: string, ...tools: string[]): CoverageEntry => ({ kind: "internal", tools, reason });
const interaction = (reason: string, ...tools: string[]): CoverageEntry => ({ kind: "interaction", tools, reason });

export const INTERFACE_CATALOGUE = {
  subspace: exposed("drowse_list_manifolds", "drowse_get_manifold", "drowse_edit_steer_rack", "drowse_attach_probe"),
  manifolds: exposed("drowse_list_manifolds", "drowse_get_manifold", "drowse_edit_steer_rack", "drowse_attach_probe"),
  manifold_builder: exposed("drowse_extract_profile", "drowse_create_manifold", "drowse_discover_manifold", "drowse_generate_manifold", "drowse_fit_manifold", "drowse_read_form", "drowse_edit_form"),
  surface_geometry: exposed("drowse_inspect_surface"),
  manifold_merge: exposed("drowse_merge_manifolds"),
  manifold_pack: exposed("drowse_list_manifolds", "drowse_search_manifolds", "drowse_install_manifold", "drowse_list_manifold_packs", "drowse_delete_manifold_pack", "drowse_import_manifold_pack", "drowse_export_manifold_pack"),
  save_conversation: exposed("drowse_save_chat", "drowse_update_chat", "drowse_read_form", "drowse_edit_form"),
  download_chat: exposed("drowse_export_chat"),
  load_conversation: exposed("drowse_list_chats", "drowse_open_chat", "drowse_import_chat"),
  compare: exposed("drowse_compare_generations"),
  system_prompt: exposed("drowse_set_system_prompt"),
  token_drilldown: exposed("drowse_read_tree", "drowse_token_readout", "drowse_fork_token", "drowse_set_token_inspector"),
  correlation: exposed("drowse_correlate_profiles", "drowse_compare_profiles"),
  probe_inspector: exposed("drowse_probe_geometry", "drowse_read_workspace"),
  advanced_sampling: exposed("drowse_set_sampling", "drowse_explain_control"),
  health: exposed("drowse_health", "drowse_get_state"),
  appearance: exposed("drowse_set_appearance"),
  session_admin: interaction("Session inventory is read-only; bearer-key entry stays in the existing memory-only credential UI. The dashboard targets the default session.", "drowse_list_sessions", "drowse_get_state", "drowse_navigate"),
  local_runtime: interaction("Python companion installation or credential entry uses the existing setup UI; hosted model lifecycle has structured tools.", "drowse_get_state", "drowse_navigate"),
  help: exposed("drowse_explain_control", "drowse_list_actions", "drowse_navigate"),
  feedback: exposed("drowse_contact_draft", "drowse_read_contact", "drowse_send_contact"),
  node_compare: exposed("drowse_compare_nodes", "drowse_joint_logprobs"),
  transcript: exposed("drowse_export_transcript", "drowse_load_transcript"),
  template_lab: exposed("drowse_list_templates", "drowse_get_template", "drowse_create_template", "drowse_update_template", "drowse_delete_template", "drowse_score_template", "drowse_template_manifold", "drowse_read_form", "drowse_edit_form"),
  cast: exposed("drowse_set_cast_recipe", "drowse_set_role_labels", "drowse_read_workspace"),
} satisfies Record<DrawerName, CoverageEntry>;

export const INTERFACE_FAMILIES: Record<string, CoverageEntry> = {
  public_pages: exposed("drowse_about", "drowse_open_page", "drowse_set_theme"),
  agent_context: exposed("drowse_get_state", "drowse_list_actions", "drowse_select_tool_group", "drowse_explain_control", "drowse_read_result"),
  operation_jobs: exposed("drowse_get_job", "drowse_cancel_job", "drowse_reconcile_job"),
  models_and_device: exposed("drowse_list_models", "drowse_check_device", "drowse_select_model", "drowse_download_model", "drowse_load_model", "drowse_unload_model", "drowse_list_sessions", "drowse_health"),
  model_storage: exposed("drowse_read_storage", "drowse_protect_storage", "drowse_delete_local_data", "drowse_runtime_recovery"),
  model_instrument_packs: exposed("drowse_download_pack", "drowse_activate_instrument", "drowse_instrument_sources"),
  workspace_navigation: exposed("drowse_navigate", "drowse_read_workspace", "drowse_set_loom_view", "drowse_set_token_inspector"),
  visible_forms: exposed("drowse_read_form", "drowse_edit_form"),
  analysis_display: exposed("drowse_set_analysis_view"),
  pending_actions: exposed("drowse_cancel_pending_action"),
  file_transfers: exposed("drowse_begin_file", "drowse_write_file", "drowse_finish_file", "drowse_read_file", "drowse_download_file", "drowse_delete_file"),
  composer: exposed("drowse_set_composer", "drowse_start_generation"),
  raw_buffer: exposed("drowse_set_raw_buffer", "drowse_raw_continue"),
  generation_and_comparison: exposed("drowse_start_generation", "drowse_set_comparison", "drowse_compare_generations", "drowse_fork_token"),
  sampling_and_identity: exposed("drowse_set_sampling", "drowse_set_system_prompt", "drowse_set_role_labels", "drowse_set_cast_recipe"),
  steering: exposed("drowse_set_steering", "drowse_edit_steer_rack", "drowse_list_manifolds", "drowse_get_profile"),
  probes_and_highlights: exposed("drowse_attach_probe", "drowse_detach_probe", "drowse_set_highlighting", "drowse_probe_geometry"),
  measurements_and_sources: exposed("drowse_token_readout", "drowse_set_instrument_live", "drowse_instrument_sources", "drowse_use_lens_source", "drowse_validate_lens_token", "drowse_validate_sae_feature", "drowse_sae_feature_metadata"),
  instrument_preparations: exposed("drowse_prepare_instrument", "drowse_preparation_status", "drowse_cancel_preparation"),
  artifact_authoring: exposed("drowse_extract_profile", "drowse_create_manifold", "drowse_discover_manifold", "drowse_generate_manifold", "drowse_fit_manifold", "drowse_merge_manifolds"),
  artifact_distribution: exposed("drowse_list_manifold_packs", "drowse_search_manifolds", "drowse_install_manifold", "drowse_delete_manifold", "drowse_delete_manifold_pack", "drowse_import_manifold_pack", "drowse_export_manifold_pack"),
  templates: exposed("drowse_list_templates", "drowse_get_template", "drowse_create_template", "drowse_update_template", "drowse_delete_template", "drowse_score_template", "drowse_template_manifold"),
  conversation_tree: exposed("drowse_read_tree", "drowse_navigate_tree", "drowse_edit_node", "drowse_branch_node", "drowse_delete_node", "drowse_star_node", "drowse_annotate_node", "drowse_filter_tree", "drowse_edge_label"),
  comparison_evidence: exposed("drowse_compare_profiles", "drowse_correlate_profiles", "drowse_compare_nodes", "drowse_joint_logprobs"),
  transcripts: exposed("drowse_export_transcript", "drowse_load_transcript"),
  conversation_library: exposed("drowse_list_chats", "drowse_save_chat", "drowse_update_chat", "drowse_open_chat", "drowse_duplicate_chat", "drowse_delete_chat", "drowse_export_chat", "drowse_import_chat"),
  appearance: exposed("drowse_set_appearance", "drowse_reset_settings"),
  contact: exposed("drowse_contact_draft", "drowse_read_contact", "drowse_send_contact"),
  app_updates: exposed("drowse_app_update"),
  credentials_and_companion: interaction("OS-level installation and memory-only bearer-key entry remain user interactions.", "drowse_navigate"),
};

type ServiceCoverage = {
  [Service in keyof typeof RUNTIME_SERVICE_METHODS]: {
    [Method in keyof (typeof RUNTIME_SERVICE_METHODS)[Service]]: CoverageEntry;
  };
};

export const RUNTIME_SERVICE_CATALOGUE = {
  sessions: {
    operationStatus: exposed("drowse_reconcile_job"),
    list: exposed("drowse_list_sessions"),
    get: exposed("drowse_get_state"),
    patch: exposed("drowse_set_sampling", "drowse_set_system_prompt"),
    validateSteering: internal("Shared preflight validates expressions before changing steering or importing a conversation.", "drowse_set_steering", "drowse_import_chat"),
  },
  profiles: {
    list: exposed("drowse_list_profiles"), get: exposed("drowse_get_profile"),
    correlation: exposed("drowse_correlate_profiles"), pairwise: exposed("drowse_compare_profiles"), extract: exposed("drowse_extract_profile"),
  },
  probes: {
    list: exposed("drowse_read_workspace"), attach: exposed("drowse_attach_probe"), detach: exposed("drowse_detach_probe"), geometry: exposed("drowse_probe_geometry"),
  },
  manifolds: {
    inspectSurface: exposed("drowse_inspect_surface"), list: exposed("drowse_list_manifolds"), get: exposed("drowse_get_manifold"),
    create: exposed("drowse_create_manifold"), createDiscover: exposed("drowse_discover_manifold"), createFromTemplate: exposed("drowse_template_manifold"),
    delete: exposed("drowse_delete_manifold"), search: exposed("drowse_search_manifolds"), install: exposed("drowse_install_manifold"),
    merge: exposed("drowse_merge_manifolds"), fit: exposed("drowse_fit_manifold"), generate: exposed("drowse_generate_manifold"),
    drowseArchiveList: exposed("drowse_list_manifold_packs"),
    drowseArchiveInstall: exposed("drowse_import_manifold_pack"),
    drowseArchiveExport: exposed("drowse_export_manifold_pack"),
    drowseArchiveDelete: exposed("drowse_delete_manifold_pack"),
  },
  templates: {
    list: exposed("drowse_list_templates"), get: exposed("drowse_get_template", "drowse_update_template"), create: exposed("drowse_create_template", "drowse_update_template"),
    delete: exposed("drowse_delete_template"), score: exposed("drowse_score_template"),
  },
  tree: {
    replayCapabilities: internal("Read before starting replay to exclude unsupported analyses.", "drowse_read_tree", "drowse_joint_logprobs"),
    get: exposed("drowse_read_tree"),
    reset: internal("Conversation replacement and new-chat lifecycle own authoritative tree reset.", "drowse_open_chat", "drowse_import_chat"),
    restore: internal("Validated saved-chat import restores the full tree and its model/session identity.", "drowse_open_chat", "drowse_import_chat"),
    active: internal("Generation and workspace snapshots use the active branch as context.", "drowse_read_workspace"),
    navigate: exposed("drowse_navigate_tree"), edit: exposed("drowse_edit_node"), branch: exposed("drowse_branch_node"),
    delete: exposed("drowse_delete_node"), star: exposed("drowse_star_node"), note: exposed("drowse_annotate_node"),
    edgeLabel: exposed("drowse_edge_label"), filter: exposed("drowse_filter_tree"), diff: exposed("drowse_compare_nodes"),
    transcriptExport: exposed("drowse_export_transcript"), transcriptLoad: exposed("drowse_load_transcript"), jointLogprobs: exposed("drowse_joint_logprobs"),
    cast: exposed("drowse_read_workspace"), castPut: exposed("drowse_set_cast_recipe"), castDelete: exposed("drowse_set_cast_recipe"),
  },
  instruments: {
    setLive: exposed("drowse_set_instrument_live"), sources: exposed("drowse_instrument_sources"),
    setLensSource: exposed("drowse_use_lens_source"), activateInstalledPack: exposed("drowse_activate_instrument"),
    startPreparation: exposed("drowse_prepare_instrument"), preparationStatus: exposed("drowse_preparation_status", "drowse_prepare_instrument"),
    cancelPreparation: exposed("drowse_cancel_preparation", "drowse_prepare_instrument"), tokenReadout: exposed("drowse_token_readout"),
    validateLensToken: exposed("drowse_validate_lens_token"), validateSaeFeature: exposed("drowse_validate_sae_feature"),
    saeFeaturesMetadata: exposed("drowse_sae_feature_metadata"),
  },
} satisfies ServiceCoverage;

export function catalogueInterfaces(): { id: DrawerName; kind: CoverageEntry["kind"]; tools: string[]; reason?: string }[] {
  return Object.entries(INTERFACE_CATALOGUE).map(([id, entry]) => ({ id: id as DrawerName, ...entry }));
}

export function catalogueServices(): { id: string; kind: CoverageEntry["kind"]; tools: string[]; reason?: string }[] {
  return Object.entries(RUNTIME_SERVICE_CATALOGUE).flatMap(([service, methods]) =>
    Object.entries(methods).map(([method, entry]) => ({ id: `${service}.${method}`, ...entry })),
  );
}
