import recording from "../data/not-found-messages.json";

export { recording };

export function chooseMessage(previousId: string | null, random = Math.random()): number {
  const candidates = recording.messages.map((_, index) => index)
    .filter(index => recording.messages.length === 1 || recording.messages[index].id !== previousId);
  return candidates[Math.min(candidates.length - 1, Math.floor(random * candidates.length))];
}

export function nextMessage(previousId: string | null = null): number {
  let previous = previousId;
  if (previous === null) {
    try { previous = sessionStorage.getItem("drowse.404.last-message"); } catch { /* Storage can be unavailable in private contexts. */ }
  }
  const index = chooseMessage(previous);
  try { sessionStorage.setItem("drowse.404.last-message", recording.messages[index].id); } catch { /* Random selection still works without persistence. */ }
  return index;
}
