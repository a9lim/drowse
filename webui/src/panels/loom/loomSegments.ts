import type { LoomNodeJSON } from "../../lib/types";

export interface LoomSegment {
  id: string;
  parentId: string | null;
  depth: number;
  node: LoomNodeJSON;
  memberIds: string[];
  start: number;
  end: number;
  shared: boolean;
}

type TokenRow = NonNullable<LoomNodeJSON["tokens"]>[number];

// Token identity keys are memoized on object identity: a saved node keeps its
// identity until the tree replaces it, and the live overlay reuses each token
// row across frames, so a streamed frame only keys the newly arrived tokens.
const nodeKeys = new WeakMap<LoomNodeJSON, string[] | null>();
const rowKeys = new WeakMap<TokenRow, { roleLabel: string | null; previous: number; key: string }>();

/** One key per model token (role label, text, and the raw ids it covers), or
 *  null when the reply has no exact token record and must never be merged. */
function tokenKeys(node: LoomNodeJSON): string[] | null {
  if (nodeKeys.has(node)) return nodeKeys.get(node)!;
  let result: string[] | null = null;
  if (node.role === "assistant" && node.recipe && node.raw_token_ids?.length && node.tokens?.length) {
    const roleLabel = node.role_label ?? null;
    let previous = -1;
    const keys: string[] = [];
    for (const token of node.tokens) {
      const index = token.raw_index;
      if (index == null || !Number.isInteger(index) || index <= previous || token.token_id == null ||
        node.raw_token_ids[index] !== token.token_id) break;
      let memo = rowKeys.get(token);
      if (!memo || memo.roleLabel !== roleLabel || memo.previous !== previous) {
        memo = { roleLabel, previous, key: JSON.stringify([roleLabel, token.text, node.raw_token_ids.slice(previous + 1, index + 1)]) };
        rowKeys.set(token, memo);
      }
      keys.push(memo.key);
      previous = index;
    }
    if (keys.length === node.tokens.length) result = keys;
  }
  nodeKeys.set(node, result);
  return result;
}

/** A display-only radix tree. Stored replies and their token indexes stay intact. */
export function segmentLoom(nodes: LoomNodeJSON[], activeIds: ReadonlySet<string>): LoomSegment[] {
  const ids = new Set(nodes.map(node => node.id));
  const children = new Map<string | null, LoomNodeJSON[]>();
  const keys = new Map<string, string[]>();
  for (const node of nodes) {
    const parent = node.parent_id && ids.has(node.parent_id) ? node.parent_id : null;
    const siblings = children.get(parent) ?? [];
    siblings.push(node);
    children.set(parent, siblings);
    const tokenKeyList = tokenKeys(node);
    if (tokenKeyList) keys.set(node.id, tokenKeyList);
  }
  const result: LoomSegment[] = [];
  type Job = { nodes: LoomNodeJSON[]; parentId: string | null; depth: number; start: number };
  const jobs: Job[] = [{ nodes: children.get(null) ?? [], parentId: null, depth: 0, start: 0 }];
  while (jobs.length) {
    const job = jobs.pop()!;
    const groups = new Map<string, LoomNodeJSON[]>();
    for (const node of job.nodes) {
      const key = JSON.stringify([node.parent_id, keys.get(node.id)?.[job.start] ?? `terminal:${node.id}`]);
      const group = groups.get(key) ?? [];
      group.push(node);
      groups.set(key, group);
    }
    const next: Job[] = [];
    for (const group of groups.values()) {
      const node = group.find(member => activeIds.has(member.id)) ?? group[0];
      if (group.length > 1) {
        let end = job.start + 1;
        const first = keys.get(node.id)!;
        while (end < first.length && group.every(member => keys.get(member.id)?.[end] === first[end])) end += 1;
        const id = `shared:${JSON.stringify([group[0].id, job.start])}`;
        result.push({ id, parentId: job.parentId, depth: job.depth, node,
          memberIds: group.map(member => member.id), start: job.start, end, shared: true });
        next.push({ nodes: group, parentId: id, depth: job.depth + 1, start: end });
      } else {
        result.push({ id: node.id, parentId: job.parentId, depth: job.depth, node,
          memberIds: [node.id], start: job.start, end: node.tokens?.length ?? 0, shared: false });
        const descendants = children.get(node.id);
        if (descendants?.length) next.push({ nodes: descendants, parentId: node.id, depth: job.depth + 1, start: 0 });
      }
    }
    jobs.push(...next.reverse());
  }
  return result;
}
