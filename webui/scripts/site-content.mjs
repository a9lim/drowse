const repository = "https://github.com/a9lim/drowse";
const source = (path) => `${repository}/blob/main/${path}`;

export const homeSummary = {
  heading: "A local workbench for mechanistic interpretability",
  intro: "Drowse is an open-source workbench for mechanistic interpretability, activation steering, and local inference with language models. Run a compatible model in your browser, inspect tokens and internal representations, and compare how an intervention changes its output. A separate Python library and server support programmatic experiments.",
  links: [
    { label: "Activation steering", href: "/learn/activation-steering", description: "Explore concept vectors, manifolds, feature steering, and probe gates." },
    { label: "Inspect language models", href: "/learn/interpretability", description: "Read token probabilities, concept probes, sparse features, and layer predictions." },
    { label: "Run a model in your browser", href: "/learn/browser-models", description: "Check WebGPU requirements, model choices, downloads, and local storage." },
    { label: "Compare experiments", href: "/learn/experiments", description: "Use branching conversations, token forks, and restricted-choice scoring." },
    { label: "Developers and browser agents", href: "/developers", description: "Discover WebMCP tools and the separate Python, OpenAI-compatible, and Ollama-compatible APIs." },
    { label: "Privacy and local data", href: "/privacy", description: "Understand what stays on your device and which actions make network requests." },
  ],
};

export const guidePages = [
  {
    path: "/learn",
    title: "Learn LLM interpretability and activation steering | Drowse",
    description: "Learn to inspect language models, steer activations, run local WebGPU inference, and compare experiments in the Drowse workbench.",
    heading: "Explore language models with Drowse",
    intro: "Drowse brings language-model generation, activation steering, and mechanistic interpretability into one workbench. These guides connect the visible controls to the experiments you can run and the limits of their measurements.",
    sections: [
      {
        heading: "Start with a model and a question",
        paragraphs: ["Open the browser app, run its device check, and choose a compatible model. A chat model follows a conversation template; a base model continues raw text. Generate a baseline before changing the prompt, sampling settings, or internal activations."],
        links: [{ label: "Browser models and requirements", href: "/learn/browser-models" }, { label: "Open Drowse", href: "/app" }],
      },
      {
        heading: "Choose an experiment",
        bullets: ["Inspect a token's alternatives, concept readings, sparse features, or layer predictions.", "Apply a compatible concept, manifold position, word direction, or SAE feature during generation.", "Branch from a turn or token and compare continuations under controlled settings.", "Score named completion choices to inspect a distribution rather than only the sampled answer."],
        links: [{ label: "Token and representation inspection", href: "/learn/interpretability" }, { label: "Activation steering guide", href: "/learn/activation-steering" }, { label: "Branching and comparison guide", href: "/learn/experiments" }],
      },
      {
        heading: "Extend the workbench",
        paragraphs: ["The hosted browser edition runs locally with WebGPU. The Python edition adds a programmatic interface, inference APIs, and instrument authoring. Visiting browser agents can use Drowse's experimental WebMCP tools when their browser supports them."],
        links: [{ label: "Developer and agent guide", href: "/developers" }, { label: "Project documentation", href: source("README.md") }],
      },
    ],
  },
  {
    path: "/learn/activation-steering",
    title: "LLM activation steering, vectors and manifolds | Drowse",
    description: "Explore activation steering without changing model weights: concept vectors, manifolds, SAE features, word directions, ablation, and probe gates.",
    heading: "Activation steering for language models",
    intro: "Activation steering changes a language model's internal activity during inference. Drowse applies adjustable interventions to compatible model layers so you can test how a concept, direction, or feature affects a continuation without modifying the model's weights.",
    sections: [
      {
        heading: "Concept vectors and fitted manifolds",
        paragraphs: ["A steering vector is a direction associated with a contrast, such as formal versus casual. Drowse represents these controls in one artifact family: the manifold. A flat one-dimensional subspace provides vector steering; larger flat subspaces and curved manifolds describe several related concepts or positions.", "Use installed controls or build a manifold from labeled examples. The browser supports generating concept examples, extracting contrasts, fitting manifolds, merging artifacts, and creating manifolds from completion templates. Fits belong to a particular model and must be compatible with the active runtime."],
        links: [{ label: "Manifold and extraction documentation", href: source("README.md#concepts-subspaces-and-manifolds") }, { label: "Browser manifold fitting source", href: source("webui/src/hosted/runtime/browserManifoldFitting.ts") }],
      },
      {
        heading: "Compose an intervention",
        paragraphs: ["The visual steering rack and expression syntax describe the same interventions. Coefficients set strength; expressions can combine directions, keep or remove a shared component, mean-ablate a concept, or target a manifold node. Phase triggers restrict an intervention to part of generation, while probe gates condition it on available readings.", "For example, the expression below combines two installed concept controls. The values illustrate syntax; useful strengths depend on the model, fitted artifact, prompt, and desired behavior."],
        code: "0.3 honest + 0.4 warm",
        links: [{ label: "Steering expression reference", href: source("README.md#steering-expressions") }],
      },
      {
        heading: "Steer words and sparse features",
        paragraphs: ["Compatible J-lens packs provide vocabulary directions for single-token words. Compatible sparse autoencoder packs expose learned feature directions. These controls share the workbench with concept steering, but their strengths and readout units differ. A feature description is an interpretation of that feature, not a guarantee about the resulting text."],
        links: [{ label: "Understand the instruments", href: "/learn/interpretability" }, { label: "Instrument implementation", href: source("webui/src/hosted/runtime/browserInstrumentRuntime.ts") }],
      },
      {
        heading: "Compare behavior before drawing conclusions",
        paragraphs: ["Start with an unsteered baseline and compare branches with the same model, prompt, parent, and sampling settings. Strong interventions can cause repetition or incoherent output. Prompt instructions, sampling settings, and activation steering affect different parts of generation; a saved recipe is not a fine-tuned model. A probe named honest does not verify factual accuracy."],
        links: [{ label: "Design a comparison", href: "/learn/experiments" }, { label: "Control semantics and observed limitations", href: source("webui/docs/webmcp.md#choosing-controls") }],
      },
    ],
  },
  {
    path: "/learn/interpretability",
    title: "LLM interpretability: tokens, SAE features and J-lens | Drowse",
    description: "Inspect token probabilities, concept probes, sparse autoencoder features, and Jacobian-lens layer predictions in Drowse's local LLM workbench.",
    heading: "Inspect tokens and internal representations",
    intro: "Drowse combines token-level inspection with instruments for language-model representations. Use the workbench to examine what the model predicted, where an activation sits relative to a fitted concept, and how compatible layer or feature readouts change across a conversation.",
    sections: [
      {
        heading: "Token probabilities and alternative continuations",
        paragraphs: ["Select a generated token to open its detail drawer. The logits view shows the chosen token and captured alternatives with log probabilities. Surprise highlighting makes unlikely generated tokens easier to spot. You can fork from a token to test a different continuation and move the inspection cursor across tokens and turns.", "Captured alternatives are a limited recorded set. Their count is separate from sampling top-k, which changes how the next token is selected. A high token probability expresses the model's preference in context; it does not establish that a statement is true."],
        links: [{ label: "Token inspection documentation", href: source("README.md#token-level-inspection") }],
      },
      {
        heading: "Concept probes and representation geometry",
        paragraphs: ["Geometry probes locate activations relative to fitted concepts, subspaces, and manifolds. The workbench exposes coordinates, nearby labels, and supported measures such as subspace fraction and manifold membership, with layer traces where available. Probe highlighting follows these measurements through generated text.", "Coordinates, distances, fractions, and assignment probabilities have different meanings. A label describes the fitted reference examples. It should not be treated as a direct measurement of a person's beliefs, a model's private intentions, or factual correctness."],
        links: [{ label: "Geometry and measurement contract", href: source("drowse/web/AGENTS.md#wire-protocol") }, { label: "Control units and applicability", href: source("webui/src/lib/webmcp/controls.ts") }],
      },
      {
        heading: "Sparse autoencoders and Jacobian-lens readouts",
        paragraphs: ["A compatible sparse autoencoder (SAE) pack exposes learned feature activations. Drowse can show feature strengths, attach probes, and apply feature steering. Available descriptions come from the matching dictionary; they remain interpretive labels.", "A compatible Jacobian-lens (J-lens) pack maps intermediate representations toward vocabulary predictions. Inspect per-layer token probabilities and an aggregate across fitted layers. The active source determines the lens used for readout and steering. Layer readouts are measurements derived from that instrument, not a transcript of hidden reasoning."],
        links: [{ label: "SAE and lens source documentation", href: source("README.md#instruments") }, { label: "Feature description provenance", href: source("webui/src/lib/saeDescriptions.ts") }],
      },
      {
        heading: "Live measurements and historical replay",
        paragraphs: ["Enable compatible instruments for live readings, or inspect historical tokens using captured measurements and supported replay. Newly attached instruments may need a replay computation. Browser users install compatible precomputed SAE and lens packs; SAE training and J-lens fitting belong to the Python edition."],
        links: [{ label: "Browser and Python boundaries", href: "/developers" }, { label: "Hosted instrument support", href: source("webui/hosted/README.md") }],
      },
    ],
  },
  {
    path: "/learn/browser-models",
    title: "Run local language models in your browser with WebGPU | Drowse",
    description: "Run compatible Gemma, Qwen, GPT-2, and Pythia models locally with Drowse. Learn about WebGPU, downloads, device checks, and browser storage.",
    heading: "Run a language model locally in your browser",
    intro: "The hosted Drowse app runs compatible language models on your device using WebGPU. Prompts, generated text, activations, and saved experiments stay in browser-local storage and memory during the normal inference workflow. No Python installation or inference server is required for this edition.",
    sections: [
      {
        heading: "Choose a chat model or a base model",
        paragraphs: ["The model catalog includes Gemma 3 and Qwen3 chat models, plus base-model entries such as GPT-2, Pythia 70M, Gemma 3 1B PT, and Qwen 3.5 2B Base. The app determines which entries can be installed and loaded for your device and current release. A listed model is not a promise that every GPU can run it.", "Chat models use a conversation template and support model-specific role and instruction behavior. Base models continue a raw text buffer. System instructions and chat-role labels do not apply to that raw-completion workflow; use a textual scaffold or compatible activation steering."],
        links: [{ label: "Open the model picker", href: "/app?choose=1" }, { label: "Model release status and limitations", href: source("README.md#hosted-browser-edition-status") }],
      },
      {
        heading: "Check WebGPU and device requirements",
        paragraphs: ["Drowse checks for a secure context, browser workers, hardware WebGPU, and usable local storage before loading. Browser version, GPU support, available memory, and model context size affect compatibility. The compute check is a prerequisite, not a full model-load stress test.", "The browser edition requires hardware WebGPU and does not provide cloud or CPU-only inference fallback. iPhone inference remains a preview with device-specific memory constraints. If a model cannot load, follow the app's device guidance or use the Python edition on a suitable machine."],
        links: [{ label: "Browser capability checks", href: source("webui/src/hosted/runtime/capabilities.ts") }, { label: "iPhone validation requirements", href: source("webui/README.md#physical-iphone-release-gate") }],
      },
      {
        heading: "Download models and optional instruments",
        paragraphs: ["Models and instrument packs are downloaded from the configured distribution providers and checked against the signed catalog and artifact hashes. Downloads support recovery and progress reporting. Compatible SAE and J-lens packs add feature and layer inspection; availability depends on the chosen model and pack.", "An installed app shell is not the same as installed model weights. Offline use depends on having the required model and runtime files stored locally. New downloads, remote artifact searches, and missing external feature descriptions need a connection."],
        links: [{ label: "Hosted distribution and storage design", href: source("webui/hosted/README.md") }],
      },
      {
        heading: "Keep a backup of your work",
        paragraphs: ["Conversations autosave in this browser. Select Protect storage where available and download backups for work you want to keep. The browser can remove local data to free space, and local saves do not sync between devices."],
        links: [{ label: "Privacy and storage details", href: "/privacy" }, { label: "Saved conversation implementation", href: source("webui/src/hosted/ui/HostedHome.svelte") }],
      },
    ],
  },
  {
    path: "/learn/experiments",
    title: "Compare LLM experiments with branching and token forks | Drowse",
    description: "Compare language-model continuations with Loom branches, token forks, activation interventions, template scoring, and saved experiment recipes.",
    heading: "Compare language-model experiments",
    intro: "Drowse's Loom records a conversation as a branching tree. Keep a baseline, change a setting, and compare continuations from a shared point. Combine those branches with token inspection and completion scoring to examine what changed under a particular intervention.",
    sections: [
      {
        heading: "Branch from a turn or a token",
        paragraphs: ["Each authored or generated turn becomes a tree node. Navigate to an earlier turn, create a branch, regenerate a continuation, or compare nodes. Stars, notes, edge labels, and filters help organize a growing experiment. Saved chats preserve the tree so you can return to alternatives.", "A token fork begins at a selected generation position. Choose a captured alternative token or supply replacement text, then continue from that prefix. This makes a local change inspectable alongside its original branch."],
        links: [{ label: "Conversation and token workflows", href: source("README.md#webui") }, { label: "Loom implementation", href: source("webui/src/hosted/runtime/browserLoom.ts") }],
      },
      {
        heading: "Change one control at a time",
        paragraphs: ["Keep the model, prompt, parent branch, seed, and sampling settings fixed when testing a steering coefficient. Compare an unsteered baseline with the intervention, then inspect both text quality and the relevant instrument readings. Save the recipe and annotate anything else that changed.", "Temperature, top-p, top-k, and repetition penalties affect token selection. System instructions affect chat context. Steering affects internal activations. Seeds help comparisons within a fixed environment; identical behavior across devices, model revisions, or runtime builds is not guaranteed. One successful example is limited evidence of a general effect."],
        links: [{ label: "Activation steering guide", href: "/learn/activation-steering" }, { label: "Generation and comparison semantics", href: source("webui/docs/webmcp.md#choosing-controls") }],
      },
      {
        heading: "Score completion choices",
        paragraphs: ["The template lab defines a slot, candidate values, and one or more contexts. Restricted-choice scoring compares the model's likelihoods for those candidates, optionally under steering. Multi-token choices expose both summed log probability and a length-normalized view, with separate probabilities across the candidate set.", "Use this when you want to compare a distribution rather than only the sampled completion. The result is conditional on the chosen candidates and contexts. A template can also supply the examples for a fitted manifold, connecting authoring and evaluation."],
        links: [{ label: "Browser template scoring source", href: source("webui/src/hosted/runtime/browserManifoldFitting.ts") }, { label: "Template format and validation", href: source("drowse/io/templates.py") }],
      },
      {
        heading: "Record and share the evidence",
        paragraphs: ["Export conversations, transcripts, or compatible manifold archives for reuse. Correlation and pairwise geometry tools help compare supported fitted directions across layers. Record the model, artifact source, prompt, settings, and comparison method with your result. Exported files may contain your experiment text, so review their contents before sharing."],
        links: [{ label: "Available analysis and export actions", href: source("webui/src/lib/webmcp/catalogue.ts") }, { label: "Python and agent workflows", href: "/developers" }],
      },
    ],
  },
  {
    path: "/developers",
    title: "Drowse for developers: WebMCP, Python and inference APIs",
    description: "Use Drowse's experimental WebMCP browser tools or the separate Python library and OpenAI-compatible, Ollama-compatible, and native inference APIs.",
    heading: "Drowse for developers and browser agents",
    intro: "Drowse exposes typed tools to visiting browser agents through experimental WebMCP. Its separate Python edition provides a library and HTTP server. Select the interface that matches where the model runs: inside the user's browser or on a machine running Drowse's Python runtime.",
    sections: [
      {
        heading: "Discover the current browser workspace",
        paragraphs: ["Open the app and load a compatible model before using the workspace example below. Start with state and action discovery, inspect a tool's schema and availability, then select its task group and discover the registered tools again. Public-page and model-onboarding tools are available before a workspace opens; steering tools require the loaded workspace. Selecting a group does not navigate or change model settings."],
        code: "drowse_get_state({})\ndrowse_list_actions({})\ndrowse_list_actions({action: \"drowse_set_steering\"})\ndrowse_select_tool_group({group: \"steering\"})\ndrowse_list_actions({group: \"steering\"})",
        bullets: ["Use drowse_explain_control to inspect units, applicability, defaults, and interactions before changing a setting.", "Groups cover chat, raw completion, steering, saved chats, appearance, models, artifacts, templates, profiles, conversation trees, analysis, instruments, files, interface controls, and authoring. Read the current group list from drowse_list_actions.", "Use drowse_list_actions with catalogue: true to inspect coverage, unavailable operations, and actions requiring a browser interaction."],
        links: [{ label: "WebMCP integration guide", href: source("webui/docs/webmcp.md") }, { label: "Tool discovery implementation", href: source("webui/src/lib/webmcp/index.ts") }],
      },
      {
        heading: "Track operations to completion",
        paragraphs: ["Tools return structured success or error results. Mutation tools accept expected_revision to reject stale edits and request_id to reconcile an uncertain invocation. Reuse a request ID only with identical input. Long operations return a job ID. Read status with drowse_get_job, or request cancellation with drowse_cancel_job.", "Job history may survive a reload in the same tab when browser storage permits. For an interrupted generation, open its original model and call drowse_reconcile_job with the job_id. This checks runtime receipts without repeating generation. It may restore the original system prompt after a comparison, while preserving newer prompt edits. Inspect the returned state and any partial results; reading status alone does not reconcile a job.", "Large-result handles can expire and do not survive reloads. Read their chunks with drowse_read_result before leaving. Preserve existing drafts and act within the user's authorization when sending a contact message or deleting their work."],
        links: [{ label: "Job and recovery implementation", href: source("webui/src/lib/webmcp/jobs.ts") }, { label: "Revision and result contracts", href: source("webui/docs/webmcp.md#discovery-and-execution") }],
      },
      {
        heading: "Understand WebMCP availability",
        paragraphs: ["Drowse registers tools through document.modelContext.registerTool when the browser exposes that experimental API. Browser and agent support are separate requirements; unsupported browsers retain the ordinary interface. The hosted website does not expose a public remote MCP inference endpoint.", "Browser tools can generate, inspect, compare, and fit compatible manifolds and templates locally. SAE training and J-lens fitting require Python; the browser consumes compatible precomputed packs. File pickers, credential entry, and some archive operations retain their existing interaction requirements."],
        links: [{ label: "Browser capability boundaries", href: source("webui/src/hosted/runtime/capabilities.ts") }, { label: "Chrome WebMCP documentation", href: "https://developer.chrome.com/docs/ai/webmcp" }],
      },
      {
        heading: "Use the Python library and local inference APIs",
        paragraphs: ["The Python import and CLI are drowse; the distribution name is drowse.ai. DrowseSession supports programmatic steering and measurement. Running drowse serve starts the dashboard and OpenAI-compatible /v1/*, Ollama-compatible /api/*, and native /drowse/v1/* routes on the configured server.", "Inspect that server's /openapi.json and /docs for its HTTP schema. These are Python-server routes, not hosted-site inference services. Configure authentication and network access for your own installation, and follow the repository's current installation instructions."],
        links: [{ label: "Python API", href: source("README.md#python-api") }, { label: "Server implementation and OpenAPI configuration", href: source("drowse/server/app.py") }, { label: "Install from source", href: source("README.md#quick-start") }],
      },
    ],
  },
  {
    path: "/privacy",
    title: "Drowse privacy: local inference, storage and network requests",
    description: "Understand Drowse's browser-local prompts and conversations, model downloads, optional feature lookups, agent access, backups, and contact submissions.",
    heading: "Privacy and your local data",
    intro: "In the hosted browser edition, language-model inference runs on your device. The inference runtime processes prompts, generated text, activation measurements, conversations, and fitted artifacts locally. Browser agents can read workspace data when given access, and their providers handle that information separately. The site also makes network requests to load software, download models, and perform specific online actions.",
    sections: [
      {
        heading: "What stays in this browser",
        paragraphs: ["Saved conversations, settings, downloaded models, and fitted artifacts use browser-local storage. The hosted inference runtime does not upload prompts or transcripts for cloud generation or application telemetry. Local saves do not create an account backup or sync across devices.", "Select Protect storage where available and download a backup of important work. Clearing site data, closing a private browsing session, or browser cleanup can remove local files. Deleting a local chat does not delete backups you previously exported or copies you shared."],
        links: [{ label: "Browser storage and saved chats", href: source("webui/src/hosted/ui/HostedHome.svelte") }, { label: "Hosted runtime design", href: source("webui/hosted/README.md") }],
      },
      {
        heading: "Downloads and external services",
        paragraphs: ["Opening Drowse requests website files from its host. The model catalog, model weights, and compatible instrument packs are downloaded from configured providers, including Hugging Face and its delivery infrastructure. Searching for remote manifold packs sends your search query to Hugging Face.", "SAE feature descriptions use bundled data when available. A missing description may trigger a Neuronpedia lookup containing the model, dictionary, and feature identifier, without your prompt or transcript. Hosting and download providers process ordinary request metadata such as network addresses. External links use the destination site's policies."],
        links: [{ label: "Configured model distribution", href: source("browser-runtime/distribution-lock.json") }, { label: "Remote manifold search", href: source("webui/src/hosted/artifacts/hfManifolds.ts") }, { label: "SAE description lookups", href: source("webui/src/lib/saeDescriptions.ts") }],
      },
      {
        heading: "Contact messages leave your device",
        paragraphs: ["Submitting the contact or feedback form sends its fields and a submission reference to Drowse's contact endpoint. This includes your message and optional reply email; chats and model data are not automatically attached. The delivery service processes the request and, when delivery is enabled and accepted, sends it to Drowse's contact inbox.", "The contact implementation uses rate limits and delivery receipts. Receipts store a content hash, status, and provider message identifier; the inbox retains delivered message contents under its own retention policy. Contact drafts remain in page memory. The form reports whether sending was confirmed."],
        links: [{ label: "Contact delivery and privacy contract", href: source("webui/contact-worker/README.md#delivery-and-privacy-contract") }, { label: "Contact Drowse", href: "/contact" }],
      },
      {
        heading: "Browser agents, exports, and Python servers",
        paragraphs: ["A browser agent with access to Drowse's WebMCP tools can read relevant workspace data and perform exposed actions. Operation records use tab-local session storage for recovery. Information that the agent reads is also subject to that agent provider's handling. Review exported files and explicitly chosen messages before sharing them.", "In the Python edition, prompts and measurements pass between the dashboard and your configured Drowse server. If that server runs on another machine, it receives the data. Its operator controls the model, storage, and access settings."],
        links: [{ label: "Agent interface and data boundaries", href: source("webui/docs/webmcp.md#integration-boundaries") }, { label: "Developer guide", href: "/developers" }],
      },
    ],
  },
  {
    path: "/about",
    title: "About Drowse, an open-source LLM interpretability workbench",
    description: "Drowse is an open-source workbench for local language models, activation steering, and mechanistic interpretability. Find source, licensing, and contributors.",
    heading: "About Drowse",
    intro: "Drowse is an open-source workbench for exploring language models through generation, activation steering, and mechanistic interpretability. It brings branching conversations, token inspection, fitted concepts, sparse autoencoder features, and Jacobian-lens readouts into a shared interface.",
    sections: [
      {
        heading: "A workbench for questions about model behavior",
        paragraphs: ["Use Drowse to compare a baseline and an intervention, inspect predicted tokens, examine fitted representation geometry, or score a set of candidate completions. It is useful for learning interpretability methods, developing experiments, and exploring model behavior interactively.", "The browser edition runs compatible models locally with WebGPU. The Python library and server support programmatic use and additional authoring workflows. Model, device, and instrument compatibility determine what is available; an inspection result or steering recipe is not a guarantee of accuracy or reliable behavior."],
        links: [{ label: "Learn the workbench", href: "/learn" }, { label: "Project overview", href: source("README.md") }],
      },
      {
        heading: "People and source code",
        paragraphs: ["Drowse's public credits identify a9lim and treetown as the team building and maintaining the project, alongside contributors and acknowledgments. The canonical source repository is a9lim/drowse on GitHub. The credits page links the team's public profiles."],
        links: [{ label: "Team and credits", href: "/credits" }, { label: "Drowse source repository", href: repository }],
      },
      {
        heading: "Research methods and citations",
        paragraphs: ["The project documents its lineage in representation engineering, manifold steering, and the Jacobian lens. For published experiments, cite the relevant upstream methods alongside the Drowse version, exact model checkpoint, and fitted artifacts used."],
        links: [{ label: "Drowse research lineage", href: source("README.md#research-lineage-and-credits") }, { label: "Representation Engineering", href: "https://arxiv.org/abs/2310.01405" }, { label: "Manifold Steering", href: "https://arxiv.org/abs/2605.05115" }, { label: "Verbalizable Representations and the Jacobian Lens", href: "https://transformer-circuits.pub/2026/workspace/index.html" }],
      },
      {
        heading: "Open-source license and contributions",
        paragraphs: ["Drowse is licensed under GNU AGPL version 3 or later. Downloaded models, instrument packs, and third-party assets have their own licenses and attribution. Consult the relevant files when using or redistributing them.", "Report a reproducible bug, propose an improvement, or contribute through the repository. Include the model, browser or Python environment, and steps needed to reproduce the issue. Use the contact page for support, research collaboration, media inquiries, or funding and sponsorship."],
        links: [{ label: "GNU AGPL license", href: source("LICENSE") }, { label: "Contributing guide", href: source("CONTRIBUTING.md") }, { label: "Report an issue", href: `${repository}/issues` }, { label: "Contact the team", href: "/contact" }],
      },
    ],
  },
];
