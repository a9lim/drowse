<script lang="ts">
  import SitePage from "./SitePage.svelte";
  import GuideContent from "./GuideContent.svelte";
  import { htmlGuides } from "../../../../scripts/site-content.mjs";

  const page = htmlGuides.find(guide => guide.path === "/privacy")!;
</script>

<SitePage title="Privacy · Drowse" heading={page.heading} lede={page.intro} markdown="/privacy.md">
  <GuideContent {page} />
</SitePage>
