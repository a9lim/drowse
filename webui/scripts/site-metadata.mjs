export const siteName = "Drowse";
export const siteDescription = "Run language models in your browser. Inspect their predictions and test how changing internal activity affects their replies with Drowse.";
export const socialImageAlt = "Drowse: See inside your model. Inspect predictions and compare alternate replies in your browser.";

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
        image: `${origin}/social/drowse.png`, isAccessibleForFree: true },
    ],
  };
  return `<link rel="canonical" href="${origin}/" />\n<meta property="og:url" content="${origin}/" />\n<script type="application/ld+json">${JSON.stringify(schema).replaceAll("<", "\\u003c")}</script>`;
}
