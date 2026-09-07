// Python-dashboard-only SAE training and J-lens fitting jobs. The hosted
// build aliases its source panels to pack-only components and never imports
// this module.

import { createPreparationSlice } from "./preparations.svelte";
import { refreshLensSources, refreshSaeSources } from "./instruments.svelte";
import { refreshProbeList } from "./probes.svelte";
import { refreshSession } from "./session.svelte";

export const saeTrain = createPreparationSlice("sae", "train", {
  label: "SAE train",
  intervalMs: 1500,
  successMessage: "SAE trained · live",
  onSettled: async () => {
    await refreshSession();
    await refreshSaeSources();
    await refreshProbeList();
  },
});

export const lensFit = createPreparationSlice("lens", "fit", {
  label: "J-lens fit",
  intervalMs: 3000,
  successMessage: "J-lens fitted · live",
  onSettled: async () => {
    await refreshSession();
    await refreshLensSources();
  },
});
