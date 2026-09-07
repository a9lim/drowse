<script lang="ts">
  import { chatAccentPalette } from "../chatAccent";
  import { savedConversationState } from "../stores/savedConversations.svelte";

  $effect(() => {
    const palette = chatAccentPalette(savedConversationState.accent);
    if (palette.id === "purple") return;
    const root = document.documentElement;
    root.dataset.chatAccent = palette.id;
    root.style.setProperty("--chat-accent-dark", palette.dark);
    root.style.setProperty("--chat-accent-light", palette.light);
    return () => {
      delete root.dataset.chatAccent;
      root.style.removeProperty("--chat-accent-dark");
      root.style.removeProperty("--chat-accent-light");
    };
  });
</script>
