<script lang="ts">
  import FluentIcon from "./FluentIcon.svelte";
</script>

<span><FluentIcon name="back" size={24} /></span>

<style>
  span { display: inline-flex; }
  span:dir(rtl) { transform: scaleX(-1); }
</style>
