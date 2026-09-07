import { MEBIBYTE } from "./contracts";

const DEFAULT_ALPHA = 0.2;
const STALL_MS = 10_000;
const MIN_ETA_MS = 3_000;
const MIN_ETA_BYTES = 8 * MEBIBYTE;

export interface EtaSnapshot {
  throughputBytesPerSecond: number | null;
  etaSeconds: [number, number] | null;
  calculating: boolean;
  stalled: boolean;
}

export function conservativeDownlinkBytesPerSecond(downlinkMbps: number): number {
  if (!Number.isFinite(downlinkMbps) || downlinkMbps <= 0) return 0;
  return (downlinkMbps * 1_000_000 * 0.75) / 8;
}

export function nextEwma(
  previous: number | null,
  sample: number,
  alpha = DEFAULT_ALPHA,
): number | null {
  if (!Number.isFinite(sample) || sample <= 0) return previous;
  if (!(alpha > 0 && alpha <= 1)) throw new RangeError("alpha must be in (0, 1]");
  return previous === null ? sample : alpha * sample + (1 - alpha) * previous;
}

export class DownloadEtaEstimator {
  readonly alpha: number;
  private readonly startingBytes: number;
  private lastObservedAt: number;
  private lastObservedBytes: number;
  private activeElapsedMs = 0;
  private lastProgressActiveMs = 0;
  private wasOnline = true;
  private throughput: number | null;

  constructor(startedAt: number, startingBytes = 0, alpha = DEFAULT_ALPHA) {
    if (!(alpha > 0 && alpha <= 1)) throw new RangeError("alpha must be in (0, 1]");
    this.alpha = alpha;
    this.startingBytes = startingBytes;
    this.lastObservedAt = startedAt;
    this.lastObservedBytes = startingBytes;
    this.throughput = null;
  }

  observe(
    cumulativeBytes: number,
    totalBytes: number,
    now: number,
    online = true,
  ): EtaSnapshot {
    if (now < this.lastObservedAt) throw new RangeError("time must be monotonic");
    if (cumulativeBytes < this.lastObservedBytes) {
      throw new RangeError("download bytes must be monotonic");
    }

    const deltaBytes = cumulativeBytes - this.lastObservedBytes;
    const deltaMs = now - this.lastObservedAt;
    if (online && this.wasOnline) this.activeElapsedMs += deltaMs;
    if (online && this.wasOnline && deltaBytes > 0 && deltaMs > 0) {
      const sample = (deltaBytes * 1000) / deltaMs;
      this.throughput = nextEwma(this.throughput, sample, this.alpha);
      this.lastProgressActiveMs = this.activeElapsedMs;
    } else if (online && !this.wasOnline && deltaBytes > 0) {
      this.lastProgressActiveMs = this.activeElapsedMs;
    }
    this.lastObservedAt = now;
    this.lastObservedBytes = cumulativeBytes;
    this.wasOnline = online;

    const stalled =
      online && this.activeElapsedMs - this.lastProgressActiveMs >= STALL_MS;
    const transferred = cumulativeBytes - this.startingBytes;
    const calculating =
      this.throughput === null ||
      (this.activeElapsedMs < MIN_ETA_MS && transferred < MIN_ETA_BYTES);
    if (!online || stalled || calculating || this.throughput === null) {
      return {
        throughputBytesPerSecond: this.throughput,
        etaSeconds: null,
        calculating,
        stalled,
      };
    }

    const remaining = Math.max(0, totalBytes - cumulativeBytes);
    const seconds = remaining / this.throughput;
    return {
      throughputBytesPerSecond: this.throughput,
      etaSeconds: roundedEtaRange(seconds),
      calculating: false,
      stalled: false,
    };
  }
}

export function roundedEtaRange(seconds: number): [number, number] {
  if (!Number.isFinite(seconds) || seconds <= 0) return [0, 0];
  const lower = roundDuration(seconds * 0.85);
  const upper = roundDuration(seconds * 1.25);
  return [Math.max(1, lower), Math.max(lower, upper)];
}

export function formatEtaRange([lowerSeconds, upperSeconds]: [number, number]): string {
  const lower = Math.max(1, lowerSeconds);
  const upper = Math.max(lower, upperSeconds);
  if (upper < 60) return formatSameUnitRange(lower, upper, 1, "second");
  if (upper < 3_600) return formatSameUnitRange(lower, upper, 60, "minute");
  if (lower >= 3_600) return formatSameUnitRange(lower, upper, 3_600, "hour");
  return `${formatUnit(lower, 60, "minute")} - ${formatUnit(upper, 3_600, "hour")}`;
}

function formatSameUnitRange(
  lowerSeconds: number,
  upperSeconds: number,
  unitSeconds: number,
  unit: "second" | "minute" | "hour",
): string {
  const lower = roundedUnit(lowerSeconds, unitSeconds);
  const upper = roundedUnit(upperSeconds, unitSeconds);
  const suffix = upper === 1 ? unit : `${unit}s`;
  return lower === upper ? `${upper} ${suffix}` : `${lower} - ${upper} ${suffix}`;
}

function formatUnit(
  seconds: number,
  unitSeconds: number,
  unit: "minute" | "hour",
): string {
  const value = roundedUnit(seconds, unitSeconds);
  return `${value} ${value === 1 ? unit : `${unit}s`}`;
}

function roundedUnit(seconds: number, unitSeconds: number): number {
  const value = seconds / unitSeconds;
  if (unitSeconds === 3_600 && value < 10) return Number(value.toFixed(1));
  return Math.max(1, Math.round(value));
}

function roundDuration(seconds: number): number {
  if (seconds < 60) return Math.ceil(seconds / 5) * 5;
  if (seconds < 600) return Math.ceil(seconds / 30) * 30;
  return Math.ceil(seconds / 60) * 60;
}
