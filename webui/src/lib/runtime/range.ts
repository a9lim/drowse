export interface RangeResponseFacts {
  status: number;
  requestedOffset: number;
  requestedEnd?: number | null;
  expectedTotalBytes: number;
  contentRange: string | null;
  contentLength: number | null;
  storedEtag: string | null;
  responseEtag: string | null;
}

export interface RangeValidation {
  action: "append" | "restart" | "reject";
  resumable: boolean;
  reason: string | null;
}

export interface ParsedContentRange {
  start: number;
  end: number;
  total: number;
}

export function parseContentRange(value: string | null): ParsedContentRange | null {
  if (!value) return null;
  const match = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(value.trim());
  if (!match) return null;
  const start = Number(match[1]);
  const end = Number(match[2]);
  const total = Number(match[3]);
  if (![start, end, total].every(Number.isSafeInteger)) return null;
  if (start < 0 || end < start || total <= end) return null;
  return { start, end, total };
}

export function validateRangeResponse(facts: RangeResponseFacts): RangeValidation {
  if (
    facts.storedEtag !== null &&
    facts.responseEtag !== null &&
    facts.storedEtag !== facts.responseEtag
  ) {
    return {
      action: "restart",
      resumable: false,
      reason: "The immutable object's HTTP validator changed",
    };
  }

  if (facts.status === 200) {
    if (facts.requestedOffset > 0 || facts.requestedEnd != null) {
      return {
        action: "restart",
        resumable: false,
        reason: "The server ignored the requested byte range",
      };
    }
    if (
      facts.contentLength !== null &&
      facts.contentLength !== facts.expectedTotalBytes
    ) {
      return {
        action: "reject",
        resumable: false,
        reason: "The full response length does not match the signed catalog",
      };
    }
    return { action: "append", resumable: false, reason: null };
  }

  if (facts.status === 206) {
    const parsed = parseContentRange(facts.contentRange);
    if (!parsed) {
      return {
        action: "restart",
        resumable: false,
        reason: "The partial response has no valid Content-Range",
      };
    }
    if (
      parsed.start !== facts.requestedOffset ||
      parsed.total !== facts.expectedTotalBytes ||
      (facts.requestedEnd != null && parsed.end !== facts.requestedEnd)
    ) {
      return {
        action: "restart",
        resumable: false,
        reason: "The partial response does not match the requested object range",
      };
    }
    const responseBytes = parsed.end - parsed.start + 1;
    if (facts.contentLength !== null && facts.contentLength !== responseBytes) {
      return {
        action: "restart",
        resumable: false,
        reason: "Content-Length and Content-Range disagree",
      };
    }
    return { action: "append", resumable: true, reason: null };
  }

  if (facts.status === 416) {
    return {
      action: "restart",
      resumable: false,
      reason: "The server rejected the stored partial range",
    };
  }

  return {
    action: "reject",
    resumable: false,
    reason: `Unexpected download response status ${facts.status}`,
  };
}
