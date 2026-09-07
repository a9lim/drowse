export const CHAT_ACCENTS = [
  { id: "purple", name: "Lavender", dark: "#c5b3ff", light: "#5b3fbf" },
  { id: "blue", name: "Sky", dark: "#a9d5ff", light: "#245c91" },
  { id: "mint", name: "Mint", dark: "#a4dfc6", light: "#23664f" },
  { id: "rose", name: "Rose", dark: "#f2b8d4", light: "#923e68" },
  { id: "peach", name: "Peach", dark: "#f3c6a5", light: "#88502c" },
  { id: "periwinkle", name: "Iris", dark: "#bfc5ff", light: "#4b50a1" },
] as const;

export type ChatAccent = typeof CHAT_ACCENTS[number]["id"];
export const DEFAULT_CHAT_ACCENT: ChatAccent = "purple";

export function isChatAccent(value: unknown): value is ChatAccent {
  return CHAT_ACCENTS.some(accent => accent.id === value);
}

export function chatAccentPalette(value: ChatAccent = DEFAULT_CHAT_ACCENT) {
  return CHAT_ACCENTS.find(accent => accent.id === value) ?? CHAT_ACCENTS[0];
}

export function chatAccentStyle(value?: ChatAccent): string {
  const palette = chatAccentPalette(value);
  return `--chat-accent-dark: ${palette.dark}; --chat-accent-light: ${palette.light};`;
}
