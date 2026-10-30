<script lang="ts">
  import SitePage from "./SitePage.svelte";
  import GuideContent from "./GuideContent.svelte";
  import { htmlGuides } from "../../../../scripts/site-content.mjs";

  const page = htmlGuides.find(guide => guide.path === "/about")!;
</script>

<SitePage title="About · Drowse" heading={page.heading} lede={page.intro} markdown="/about.md">
  <GuideContent {page} />
</SitePage>
