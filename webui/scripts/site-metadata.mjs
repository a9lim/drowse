export const siteName = "Drowse";
export const siteTitle = "Drowse | Local LLM interpretability and activation steering";
export const sourceRepository = "https://github.com/a9lim/drowse";
export const siteDescription = "Open-source, fully local LLM interpretability workbench in your browser. Inspect token predictions and steer a model's internal activity.";
export const siteAccent = "#c5b3ff";
// Browser chrome follows the page surface, not the accent: these match theme-init.js and tokens.css.
export const themeColors = { dark: "#0b0e17", light: "#f2f4f8" };
export const socialImagePath = "/social/drowse.jpg?v=shader-orb-workbench-jpeg";
export const socialImageType = "image/jpeg";
export const socialImageAlt = "Drowse in white with Interpretability Workbench underneath, beside a large glowing shader orb.";
export const projectOwner = { "@type": "Person", name: "a9lim", url: "https://github.com/a9lim" };

export function publicOrigin(value = "") {
  if (!value) return "";
  const url = new URL(value);
  if (url.protocol !== "https:" || url.username || url.password || url.pathname !== "/" || url.search || url.hash) {
    throw new Error("DROWSE_PUBLIC_ORIGIN must be an HTTPS origin, such as https://drowse.example, without a path or credentials.");
  }
  return url.origin;
}

const jsonLd = schema => `<script type="application/ld+json">${JSON.stringify(schema).replaceAll("<", "\\u003c")}</script>`;

export function discoveryMetadata(origin, { version } = {}) {
  if (!origin) return "";
  const schema = {
    "@context": "https://schema.org",
    "@graph": [
      { "@type": "WebSite", "@id": `${origin}/#website`, url: `${origin}/`, name: siteName, description: siteDescription, inLanguage: "en", publisher: projectOwner },
      { "@type": "WebApplication", "@id": `${origin}/#app`, name: siteName, url: `${origin}/`, description: siteDescription,
        applicationCategory: "DeveloperApplication", operatingSystem: "Web browser", browserRequirements: "WebGPU and a compatible device are required for local model inference.",
        ...(version ? { softwareVersion: version } : {}),
        offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
        author: projectOwner, publisher: projectOwner,
        image: `${origin}${socialImagePath}`, isAccessibleForFree: true,
        license: `${origin}/LICENSE`, sameAs: [sourceRepository],
        featureList: ["Local language model inference", "Activation steering", "Token prediction inspection", "Trait monitoring", "Branching experiments", "Compatible SAE and Jacobian lens packs"] },
    ],
  };
  return `<link rel="canonical" href="${origin}/" />\n<meta property="og:url" content="${origin}/" />\n${jsonLd(schema)}`;
}

/** Canonical, og:url and WebPage JSON-LD for a prerendered page other than the homepage. */
export function pageDiscoveryMetadata(origin, { path, title, description }) {
  if (!origin) return "";
  const url = `${origin}${path}`;
  const schema = {
    "@context": "https://schema.org",
    "@type": "WebPage", "@id": `${url}#page`, url, name: title, description, inLanguage: "en",
    isPartOf: { "@id": `${origin}/#website` }, about: { "@id": `${origin}/#app` },
  };
  return `<link rel="canonical" href="${url}" />\n<meta property="og:url" content="${url}" />\n${jsonLd(schema)}`;
}
