export function createFrameQueue<T>(
  consume: (item: T) => void,
  schedule: (callback: () => void) => number,
  cancel: (frame: number) => void,
) {
  let pending: T[] = [];
  let frame: number | null = null;
  const flush = () => {
    if (frame !== null) cancel(frame);
    frame = null;
    const batch = pending;
    pending = [];
    for (const item of batch) consume(item);
  };
  return {
    flush,
    push(item: T) {
      pending.push(item);
      if (pending.length >= 256) flush();
      else if (frame === null) frame = schedule(flush);
    },
  };
}
