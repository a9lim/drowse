// One listener for every token chip in the explorer: hovering a token lights up its variants in
// every read of the same neuron, and Chinese tokens show pinyin and an English gloss.
import { ensureGlossary, gloss, hasGlossary } from "./gloss.svelte";

const glossFor = (text: string) => gloss.dict?.[text.trim()];

export function tokenHover(root: HTMLElement) {
  const tip = document.createElement("div");
  tip.className = "zh-gloss";
  tip.setAttribute("role", "tooltip");
  tip.id = `zh-gloss-${Math.random().toString(36).slice(2, 8)}`;
  tip.hidden = true;
  document.body.append(tip);

  let linked: Element[] = [];
  let current: HTMLElement | null = null;

  function clearLinked() {
    for (const element of linked) element.classList.remove("linked");
    linked = [];
  }

  function hideTip() {
    tip.hidden = true;
    tip.classList.remove("shown");
    current?.removeAttribute("aria-describedby");
  }

  function showTip(token: HTMLElement) {
    const zh = token.dataset.zh;
    if (!zh) return hideTip();
    const layer = Number(token.closest<HTMLElement>("[data-layer]")?.dataset.layer ?? NaN);
    const ready = Number.isFinite(layer) && hasGlossary(layer);
    if (Number.isFinite(layer) && !ready) void ensureGlossary(layer).then(() => { if (current === token) showTip(token); });
    const entry = glossFor(zh);
    tip.replaceChildren();
    const word = document.createElement("span");
    word.className = "zh-word";
    word.lang = "zh";
    word.textContent = zh;
    tip.append(word);
    if (entry) {
      const pinyin = document.createElement("span");
      pinyin.className = "zh-pinyin";
      pinyin.textContent = entry[0];
      const meaning = document.createElement("span");
      meaning.className = "zh-meaning";
      meaning.textContent = entry[1];
      tip.append(pinyin, meaning);
    } else {
      const meaning = document.createElement("span");
      meaning.className = "zh-meaning muted";
      meaning.textContent = ready || !Number.isFinite(layer) ? "Not in the dictionary: likely part of a longer word." : "Loading translation…";
      tip.append(meaning);
    }
    tip.hidden = false;
    const box = token.getBoundingClientRect();
    const width = Math.min(320, window.innerWidth - 24);
    tip.style.maxWidth = `${width}px`;
    const tipBox = tip.getBoundingClientRect();
    const left = Math.max(12, Math.min(window.innerWidth - tipBox.width - 12, box.left + box.width / 2 - tipBox.width / 2));
    const above = box.top - tipBox.height - 8;
    const top = above > 8 ? above : box.bottom + 8;
    tip.style.transform = `translate(${Math.round(left)}px, ${Math.round(top + window.scrollY)}px)`;
    token.setAttribute("aria-describedby", tip.id);
    requestAnimationFrame(() => tip.classList.add("shown"));
  }

  function enter(token: HTMLElement) {
    if (token === current) return;
    leave();
    current = token;
    const key = token.dataset.stem;
    const scope = token.closest("[data-link-scope]");
    if (key && scope) {
      linked = [...scope.querySelectorAll(`.tok[data-stem="${CSS.escape(key)}"]`)];
      if (linked.length > 1) for (const element of linked) element.classList.add("linked");
      else linked = [];
    }
    showTip(token);
  }

  function leave() {
    clearLinked();
    hideTip();
    current = null;
  }

  const over = (event: Event) => {
    const token = (event.target as Element | null)?.closest?.<HTMLElement>(".tok, .zh-term");
    if (token && root.contains(token)) enter(token);
    else if (current) leave();
  };
  const out = (event: PointerEvent | FocusEvent) => {
    const next = event.relatedTarget as Element | null;
    // The pointer may move onto the gloss itself to read it (WCAG 1.4.13).
    if (next && tip.contains(next)) return;
    if (!next || !current?.contains(next)) {
      const nextToken = next?.closest?.<HTMLElement>(".tok, .zh-term");
      if (!nextToken) leave();
    }
  };
  // Touch: a tap shows the gloss; tapping elsewhere hides it.
  const down = (event: PointerEvent) => {
    if (event.pointerType === "mouse" || tip.contains(event.target as Node)) return;
    const token = (event.target as Element | null)?.closest?.<HTMLElement>(".tok, .zh-term");
    if (token && root.contains(token)) enter(token);
    else leave();
  };
  const key = (event: KeyboardEvent) => { if (event.key === "Escape") leave(); };
  const scroll = () => { if (current) leave(); };

  root.addEventListener("pointerover", over);
  root.addEventListener("pointerout", out);
  root.addEventListener("focusin", over);
  root.addEventListener("focusout", out);
  const tipLeave = (event: PointerEvent) => {
    const next = event.relatedTarget as Element | null;
    if (!next || !current?.contains(next)) leave();
  };
  tip.addEventListener("pointerleave", tipLeave);
  document.addEventListener("pointerdown", down);
  document.addEventListener("keydown", key);
  window.addEventListener("scroll", scroll, { passive: true });

  return {
    destroy() {
      root.removeEventListener("pointerover", over);
      root.removeEventListener("pointerout", out);
      root.removeEventListener("focusin", over);
      root.removeEventListener("focusout", out);
      tip.removeEventListener("pointerleave", tipLeave);
      document.removeEventListener("pointerdown", down);
      document.removeEventListener("keydown", key);
      window.removeEventListener("scroll", scroll);
      tip.remove();
    },
  };
}

