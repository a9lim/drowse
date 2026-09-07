export function completionAnchor(
  turns: readonly { nodeId?: string | null; text?: string | null }[],
  rootId: string,
  offset: number,
): { parentNodeId: string; branch?: { nodeId: string; text: string } } {
  let parentNodeId = rootId;
  let remaining = Math.max(0, offset);
  for (const turn of turns) {
    if (remaining === 0) break;
    const text = turn.text ?? "";
    if (!turn.nodeId) throw new Error("The selected text is not saved yet.");
    if (remaining < text.length) {
      return { parentNodeId, branch: { nodeId: turn.nodeId, text: text.slice(0, remaining) } };
    }
    remaining -= text.length;
    parentNodeId = turn.nodeId;
  }
  return { parentNodeId };
}
