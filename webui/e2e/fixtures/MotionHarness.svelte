<script lang="ts">
  import MorphText from "../../src/lib/ui/MorphText.svelte";
  import RollingNumber from "../../src/lib/ui/RollingNumber.svelte";
  import Slider from "../../src/lib/Slider.svelte";
  import NumberInput from "../../src/lib/NumberInput.svelte";
  import LayerStrip from "../../src/panels/rack/LayerStrip.svelte";
  import TraceDetail from "../../src/lib/charts/TraceDetail.svelte";
  import TemplatePreview from "../../src/lib/ui/TemplatePreview.svelte";
  import ManifoldMiniMap from "../../src/panels/manifold/ManifoldMiniMap.svelte";
  let text = $state("Ready");
  let value = $state<number | null>(12);
  let slider = $state(.5);
  let changes = $state(0);
  let identity = $state("one");
  let rtl = $state(false);
  let multiline = $state(false);
  const cells = [{layer: 1, value: .1, title: "Layer 1 · 0.1"}, {layer: 5, value: .8, title: "Layer 5 · 0.8"}, {layer: 9, value: null, title: "Layer 9 · unavailable"}];
</script>
<section data-testid="motion-harness" dir={rtl ? "rtl" : "ltr"}>
  <input aria-label="Text to display" bind:value={text} />
  <button type="button" onclick={() => text = "−0.01 → +0.01"}>Signed</button>
  <button type="button" onclick={() => { text = "Latest 10000"; identity = "two"; }}>New identity</button>
  <button type="button" onclick={() => rtl = !rtl}>RTL</button>
  <button type="button" onclick={() => multiline = !multiline}>Wrap</button>
  <p data-testid="display" style:max-width={multiline ? "8ch" : "100%"}><MorphText {text} {identity} /></p>
  <p data-testid="number"><RollingNumber value={value ?? NaN} digits={2} signed /></p>
  <div class="clip"><Slider bind:value={slider} min={0} max={1} step={.01} ariaLabel="Harness slider" /></div>
  <NumberInput bind:value min={-100} max={10000} allowEmpty ariaLabel="Harness number" onchange={() => changes++} />
  <p data-testid="commits">{changes}</p>
  <LayerStrip {cells} scale={1} ariaLabel="Harness layers" />
  <TraceDetail points={[.1, .7, null, -.2]} />
  <TemplatePreview sentence="Today is [DAY]." slot="[DAY]" values={["Monday", "Tuesday"]} />
  <ManifoldMiniMap info={{family:"geometry", name:"example", manifold:"local/example", top_n:2, layers:[1], node_labels:["calm", "alert"], node_count:2, domain:{type:"box", axes:[{name:"x",lo:-1,hi:1},{name:"y",lo:-1,hi:1}]}, intrinsic_dim:2, feature_space:"residual", is_affine:true, node_coords:[[-1,0],[1,0]]}} trajectory={[]} settled={null} />
  <div style="height:1200px"></div>
  <p data-testid="offscreen"><MorphText {text} /></p>
</section>
<style>
  section { padding: 16px; background: var(--bg); color: var(--fg); max-width: 520px; margin: auto; }
  input, button { min-height: 44px; max-width: 100%; }
  .clip { overflow: hidden; border-radius: 8px; }
</style>
