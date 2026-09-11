import type { DrowseWebLlmRuntime } from "../src/hosted/runtime/webLlmEngine";
import type { WebLlmGenerationInput } from "../src/hosted/runtime/webLlmGeneration";

type ChatMessages = Extract<WebLlmGenerationInput, { kind: "chat" }>["messages"];

export async function benchmarkGeneration(runtime: DrowseWebLlmRuntime, maxTokens: number) {
  const startedAt = performance.now();
  const engine = (runtime as any).requireEngine();
  const records = [];
  const comparisons = [];
  const decodeRecords = [];
  const prompt = "Read this background: " +
    "Water evaporates from oceans, condenses into clouds, and falls as rain. ".repeat(12) +
    "\nReply with only the word OK.";
  const initialMessages = [{ role: "user" as const, content: prompt }];
  async function generate(messages: ChatMessages, temperature: number, topK = 50) {
    const start = performance.now();
    let firstTokenMs: number | null = null;
    const ids: (number | null)[] = [];
    const result = await runtime.streamGeneration({
      input: { kind: "chat", messages },
      thinking: false, generationSeat: "assistant", generationRoleName: null,
      hookProgram: null,
      sampling: { max_tokens: maxTokens, seed: 17, temperature, top_k: topK, top_p: 0.9, return_top_k: 8 },
    }, token => {
      firstTokenMs ??= performance.now() - start;
      ids.push(token.tokenId);
    });
    return { firstTokenMs, elapsedMs: performance.now() - start, ids, ...result };
  }
  await generate(initialMessages, 0);
  for (const temperature of [0, 0.7]) {
    for (let repeat = 0; repeat < 3; repeat++) {
      const modes = repeat % 2 === 0 ? ["reset", "reuse"] : ["reuse", "reset"];
      const pair: Record<string, Awaited<ReturnType<typeof generate>>> = {};
      for (const mode of modes) {
        await engine.resetChat();
        const first = await generate(initialMessages, temperature);
        const messages: ChatMessages = [
          ...initialMessages,
          { role: "assistant", content: first.text },
          { role: "user", content: "Explain the next stage in the water cycle." },
        ];
        if (mode === "reset") await engine.resetChat();
        const result = await generate(messages, temperature);
        pair[mode] = result;
        records.push({ mode, temperature, repeat, firstTerminalReason: first.terminalReason,
          firstIds: first.ids, firstText: first.text, ...result });
      }
      comparisons.push({ temperature, repeat,
        tokensMatch: JSON.stringify(pair.reset.ids) === JSON.stringify(pair.reuse.ids) });
    }
  }
  for (let repeat = 0; repeat < 3; repeat++) {
    await engine.resetChat();
    const result = await generate([{ role: "user", content:
      "If you had to choose one animal to be for the rest of your life, which would it be?" }], 1, 0);
    decodeRecords.push({ repeat, temperature: 1, topK: 0, ...result });
  }
  if (comparisons.some(comparison => !comparison.tokensMatch)) {
    throw new Error("Conversation prefix reuse changed the generated token sequence");
  }
  return { startedAt, records, comparisons, decodeRecords };
}
