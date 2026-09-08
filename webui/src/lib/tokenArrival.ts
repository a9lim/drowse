export function createTokenArrival() {
  const arrivals = new WeakMap<object, number>();
  const streams = new Map<string, { length: number; last: object | undefined }>();
  function track(tokens: object[], active: boolean, channel: string) {
    const previous = streams.get(channel);
    const continued = previous && (previous.length === 0 || tokens[previous.length - 1] === previous.last);
    const start = continued ? previous.length : 0;
    const time = previous && active ? performance.now() : -Infinity;
    for (let index = start; index < tokens.length; index++) {
      const token = tokens[index];
      if (!arrivals.has(token)) arrivals.set(token, continued ? time : -Infinity);
    }
    streams.set(channel, { length: tokens.length, last: tokens.at(-1) });
  }
  function reveal(node: HTMLElement, token: object | null) {
    const born = token ? arrivals.get(token) : undefined;
    if (token) arrivals.delete(token);
    if (born === undefined || performance.now() - born > 140 || matchMedia("(prefers-reduced-motion: reduce)").matches
      || !document.getSelection()?.isCollapsed) return;
    const animation = node.animate([{ opacity: .45 }, { opacity: 1 }], { duration: 120, easing: "ease-out" });
    return { destroy() { animation.cancel(); } };
  }
  return { track, reveal };
}
