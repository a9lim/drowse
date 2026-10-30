import baselinePrompts from "../../../drowse/data/baseline_prompts.json";

export const EXAMPLES_PER_ROUND = baselinePrompts.length;
export const EXAMPLE_BUDGETS = [
  { rounds: 1, label: "Standard", description: "The fastest full set of prompts." },
  { rounds: 2, label: "More examples", description: "Two responses per prompt · about 2× the generation time." },
  { rounds: 4, label: "Thorough", description: "Four responses per prompt · about 4× the generation time." },
] as const;
