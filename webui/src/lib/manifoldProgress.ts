export interface WorkProgress {
  stage: string;
  label: string;
  completed: number | null;
  total: number | null;
  detail: string;
}

const FIT_STAGES: Record<string, string> = {
  capturing: "Reading examples",
  committing: "Saving readings",
  pooling: "Combining readings",
  fitting: "Fitting the concept",
  topology: "Finding the shape",
  surface: "Fitting the surface",
  covariance: "Measuring variation",
  sigma: "Calibrating the scale",
  origin: "Setting the baseline",
};

export function readWorkProgress(data: unknown, concepts: readonly string[] = []): WorkProgress | null {
  if (!data || typeof data !== "object") return null;
  const value = data as Record<string, unknown>;
  const message = typeof value.message === "string" ? value.message : "";
  const generation = message.match(/^Generating ["'](.+?)["'] responses? (\d+)(?:-\d+)?\/(\d+)/);
  if (generation) {
    const [, concept, position, count] = generation;
    const index = concepts.indexOf(concept);
    const perConcept = Number(count);
    const total = index < 0 ? perConcept : perConcept * concepts.length;
    const completed = (index < 0 ? 0 : index * perConcept) + Number(position) - 1;
    return {
      stage: "generating", label: "Generating examples",
      completed: Math.max(0, Math.min(total, completed)), total,
      detail: `${concept}: example ${position} of ${count}`,
    };
  }
  if (typeof value.stage === "string" && value.stage in FIT_STAGES) {
    const total = typeof value.total === "number" && Number.isFinite(value.total) && value.total > 0 ? value.total : null;
    const completed = total !== null && typeof value.completed === "number" && Number.isFinite(value.completed)
      ? Math.max(0, Math.min(total, value.completed)) : null;
    return {
      stage: value.stage, label: FIT_STAGES[value.stage], completed, total,
      detail: typeof value.layer === "number" ? `Layer ${value.layer}` : "",
    };
  }
  return message ? { stage: "working", label: "Fitting the concept", completed: null, total: null, detail: message } : null;
}

export interface WorkSample { at: number; completed: number }

export function remainingWorkMs(samples: readonly WorkSample[], total: number | null, now: number): number | null {
  const first = samples[0];
  const last = samples.at(-1);
  if (!first || !last || total === null) return null;
  const done = last.completed - first.completed;
  const elapsed = last.at - first.at;
  if (done < 3 || elapsed < 2000 || last.completed >= total) return null;
  const perUnit = elapsed / done;
  // An unusually long current step invalidates the old estimate.
  if (now - last.at > Math.max(15000, perUnit * 3)) return null;
  return Math.max(perUnit, (total - last.completed) * perUnit - (now - last.at));
}

export function elapsedLabel(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(seconds / 60);
  return minutes ? `${minutes}m ${String(seconds % 60).padStart(2, "0")}s` : `${seconds}s`;
}

export function etaLabel(ms: number): string {
  const seconds = Math.max(5, Math.ceil(ms / 5000) * 5);
  if (seconds < 60) return `About ${seconds} seconds left`;
  const minutes = Math.ceil(seconds / 60);
  return `About ${minutes} ${minutes === 1 ? "minute" : "minutes"} left`;
}
