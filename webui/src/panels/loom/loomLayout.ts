export interface LoomLayoutInput {
  id: string;
  parentId: string | null;
  depth: number;
  height?: number;
  width?: number;
}

export interface LoomLayoutNode {
  id: string;
  parentId: string | null;
  depth: number;
  x: number;
  y: number;
  height: number;
  width: number;
}

export interface LoomLayoutEdge {
  parentId: string;
  childId: string;
  path: string;
}

export interface LoomLayout {
  nodes: LoomLayoutNode[];
  edges: LoomLayoutEdge[];
  width: number;
  height: number;
}

export const LOOM_NODE_WIDTH = 276;
export const LOOM_NODE_HEIGHT = 168;
export const LOOM_COLUMN_GAP = 112;

const CANVAS_PADDING = 28;
const ROW_GAP = 44;

export function loomMessageWidth(text: string, tokenCount = 0): number {
  const size = text.length + tokenCount * 6;
  return size > 900 ? 680 : size > 450 ? 552 : size > 180 ? 414 : LOOM_NODE_WIDTH;
}

export function layoutLoom(inputs: LoomLayoutInput[]): LoomLayout {
  if (inputs.length === 0) {
    return { nodes: [], edges: [], width: 0, height: 0 };
  }

  const byId = new Map(inputs.map((input) => [input.id, input]));
  const children = new Map<string, string[]>();
  const roots: string[] = [];

  for (const input of inputs) {
    if (input.parentId && byId.has(input.parentId)) {
      const siblings = children.get(input.parentId) ?? [];
      siblings.push(input.id);
      children.set(input.parentId, siblings);
    } else {
      roots.push(input.id);
    }
  }

  const heightFor = (id: string): number => {
    const height = byId.get(id)?.height;
    return typeof height === "number" && Number.isFinite(height) && height > 0
      ? height
      : LOOM_NODE_HEIGHT;
  };
  let nextLeafTop = CANVAS_PADDING;
  const centerY = new Map<string, number>();
  const visiting = new Set<string>();

  function place(root: string): void {
    const stack = [{ id: root, expanded: false }];
    while (stack.length > 0) {
      const { id, expanded } = stack.pop()!;
      const childIds = children.get(id) ?? [];
      if (expanded) {
        centerY.set(id, (centerY.get(childIds[0])! + centerY.get(childIds.at(-1)!)!) / 2);
        visiting.delete(id);
      } else if (!centerY.has(id)) {
        if (childIds.length === 0 || visiting.has(id)) {
          const height = heightFor(id);
          centerY.set(id, nextLeafTop + height / 2);
          nextLeafTop += height + ROW_GAP;
          continue;
        }
        visiting.add(id);
        stack.push({ id, expanded: true });
        for (let index = childIds.length - 1; index >= 0; index -= 1) {
          stack.push({ id: childIds[index], expanded: false });
        }
      }
    }
  }

  for (const root of roots) place(root);
  for (const input of inputs) place(input.id);

  const minDepth = inputs.reduce((minimum, input) => Math.min(minimum, input.depth), Infinity);
  const widths = new Map(inputs.map(input => [input.id,
    typeof input.width === "number" && Number.isFinite(input.width) && input.width > 0 ? input.width : LOOM_NODE_WIDTH,
  ]));
  const columns = new Map<number, number>();
  for (const input of inputs) columns.set(input.depth, Math.max(columns.get(input.depth) ?? 0, widths.get(input.id)!));
  const columnX = new Map<number, number>();
  let nextColumnX = CANVAS_PADDING;
  for (const depth of [...columns.keys()].sort((a, b) => a - b)) {
    columnX.set(depth, nextColumnX);
    nextColumnX += columns.get(depth)! + LOOM_COLUMN_GAP;
  }
  const placedNodes = inputs.map<LoomLayoutNode>((input) => {
    const height = heightFor(input.id);
    return {
      ...input,
      height,
      width: widths.get(input.id)!,
      depth: input.depth - minDepth,
      x: columnX.get(input.depth)!,
      y: centerY.get(input.id)! - height / 2,
    };
  });
  const minY = placedNodes.reduce((minimum, node) => Math.min(minimum, node.y), Infinity);
  const yShift = Math.max(0, CANVAS_PADDING - minY);
  const nodes = yShift === 0
    ? placedNodes
    : placedNodes.map((node) => ({ ...node, y: node.y + yShift }));
  const positioned = new Map(nodes.map((node) => [node.id, node]));
  const edges: LoomLayoutEdge[] = [];

  for (const child of nodes) {
    if (!child.parentId) continue;
    const parent = positioned.get(child.parentId);
    if (!parent) continue;
    const startX = parent.x + parent.width;
    const startY = parent.y + parent.height / 2;
    const endX = child.x;
    const endY = child.y + child.height / 2;
    const bendX = startX + (endX - startX) * 0.5;
    edges.push({
      parentId: parent.id,
      childId: child.id,
      path: `M ${startX} ${startY} C ${bendX} ${startY}, ${bendX} ${endY}, ${endX} ${endY}`,
    });
  }

  const maxX = nodes.reduce((maximum, node) => Math.max(maximum, node.x + node.width), 0);
  const maxY = nodes.reduce((maximum, node) => Math.max(maximum, node.y + node.height), 0);
  return {
    nodes,
    edges,
    width: maxX + CANVAS_PADDING,
    height: maxY + CANVAS_PADDING,
  };
}

export function loomBranchJunctions(graph: LoomLayout): Array<{ id: string; x: number; y: number }> {
  const childCounts = new Map<string, number>();
  for (const edge of graph.edges) {
    childCounts.set(edge.parentId, (childCounts.get(edge.parentId) ?? 0) + 1);
  }
  return graph.nodes.filter((node) => (childCounts.get(node.id) ?? 0) > 1).map((node) => ({
    id: node.id,
    x: node.x + node.width + LOOM_COLUMN_GAP * 0.52,
    y: node.y + node.height / 2,
  }));
}
