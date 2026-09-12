export const SAMPLING_HELP = {
  temperature: "Temperature controls randomness. Higher values vary the token choices. Lower values make them more predictable.",
  advancedTemperature: "Temperature controls randomness. Enter a value above the slider's usual range here. Zero selects the most likely token.",
  topP: "Top P limits sampling to the smallest set of tokens whose probabilities add up to this value. Lower values narrow the choices.",
  maxTokens: "Max tokens is the maximum number of tokens the model may generate. A token is usually a word or part of a word.",
  thinking: "When supported, Thinking lets the model use a separate reasoning phase before it writes the visible reply.",
  topK: "Top K limits sampling to the K most likely next tokens, up to the full model vocabulary. Zero disables this filter. Larger pools can be slower. Leave it blank to use Drowse's default of 1,024.",
  frequencyPenalty: "Frequency penalty reduces the probability of tokens in proportion to how often they have already appeared.",
  presencePenalty: "Presence penalty reduces the probability of any token that has already appeared, regardless of frequency.",
  returnTopK: "Return top K retains alternative tokens from the sampling pool so you can inspect or branch from them later. Zero disables recording alternatives; it does not disable the sampling filter. Large counts increase memory use and saved-chat size. Browser instrument readouts remain limited to eight entries.",
  seed: "Seed fixes the random-number sequence so repeated runs are easier to compare. Values up to JavaScript's safe-integer limit (9,007,199,254,740,991) are retained; the Python server also accepts negative safe integers. Leave it blank for a new seed. Generating multiple siblings derives 31-bit branch seeds. Token-fork seed overrides also use 0–2,147,483,647. Matching seeds do not guarantee identical output across devices or runtimes.",
} as const;

export const IDENTITY_HELP = {
  chat: "Chat name, avatar, and accent identify the saved conversation. They do not change model behavior or its chat-template role labels.",
  role: "Role labels replace supported chat-template role names and can affect output. They are not display names or a guaranteed instruction to adopt a persona.",
  cast: "Standing defaults used only when a generation leaves that setting unset. The visible composer sends its rack expression and thinking setting explicitly, so those override cast defaults even when empty or off. A cast seed can apply when the visible seed is not fixed. Private notes are metadata, not model instructions.",
  systemPrompt: "Natural-language instructions used by new generations in this session. An empty string clears them. Base models use raw text instead of system messages.",
} as const;
