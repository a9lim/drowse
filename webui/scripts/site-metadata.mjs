export const siteName = "Drowse";
export const siteTitle = "Drowse | Local LLM interpretability and activation steering";
export const sourceRepository = "https://github.com/a9lim/drowse";
export const siteDescription = "Drowse is an open-source and fully local AI mechanistic interpretability workbench that works fully in your browser. Use Drowse to research large language models, inspect predictions and change an LLMs internal activity.";
export const siteAccent = "#c5b3ff";
export const socialImagePath = "/social/drowse.png?v=shader-orb-workbench";
export const socialImageAlt = "Drowse in white with Interpretability Workbench underneath, beside a large glowing shader orb.";

export function publicOrigin(value = "") {
  if (!value) return "";
  const url = new URL(value);
  if (url.protocol !== "https:" || url.username || url.password || url.pathname !== "/" || url.search || url.hash) {
    throw new Error("DROWSE_PUBLIC_ORIGIN must be an HTTPS origin, such as https://drowse.example, without a path or credentials.");
  }
  return url.origin;
}

export function discoveryMetadata(origin) {
  if (!origin) return "";
  const schema = {
    "@context": "https://schema.org",
    "@graph": [
      { "@type": "WebSite", "@id": `${origin}/#website`, url: `${origin}/`, name: siteName, description: siteDescription, inLanguage: "en" },
      { "@type": "WebApplication", "@id": `${origin}/#app`, name: siteName, url: `${origin}/app`, description: siteDescription,
        applicationCategory: "DeveloperApplication", operatingSystem: "Web browser", browserRequirements: "WebGPU and a compatible device are required for local model inference.",
        image: `${origin}${socialImagePath}`, isAccessibleForFree: true,
        license: `${origin}/LICENSE`, sameAs: [sourceRepository],
        featureList: ["Local language model inference", "Activation steering", "Token prediction inspection", "Trait monitoring", "Branching experiments", "Compatible SAE and Jacobian lens packs"] },
    ],
  };
  return `<link rel="canonical" href="${origin}/" />\n<meta property="og:url" content="${origin}/" />\n<script type="application/ld+json">${JSON.stringify(schema).replaceAll("<", "\\u003c")}</script>`;
}
