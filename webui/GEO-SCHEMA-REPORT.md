# Drowse metadata and structured data

Updated: 2026-09-04. Scope: implementation validation, not a prediction of rankings or AI citations.

## Implemented

- Static page title: Drowse. A description matching the existing local-inference, token-inspection, steering, and branching features.
- Open Graph and Twitter large-image metadata, including descriptive image alt text and a 1200 × 630 PNG. No private chat names, prompts, or model output enter share metadata or tab titles.
- WebSite and WebApplication JSON-LD in the built HTML, with absolute identifiers, canonical URL, and a homepage-only sitemap when the public origin is configured.
- A stable purple P favicon for discovery and installed-app icons; separate browser-tab state badges do not change the site's public identity.
- Preview builds remain noindex. Release preparation allows the public homepage while retaining noindex headers on `/app` and `/app/*`.

## Public origin required

Set `DROWSE_PUBLIC_ORIGIN` to the confirmed HTTPS origin in the build environment. It must have no path, query, or credentials. Without it, previews use a local social-image path and omit canonical URLs, sitemap, and JSON-LD instead of publishing invented URLs. Release preparation fails with an actionable message if the origin was not supplied.

The public domain has not yet been confirmed for this change. No deployment or live social-crawler validation was performed.

## Validation and deliberate omissions

Tests cover metadata escaping and origin validation, JSON parsing, declared image dimensions, static HTML metadata, favicon states and lifecycle, route titles, and reduced motion. Production URLs still require validation after deployment.

The structured-data skill was applied selectively: no fabricated organization/person details, ratings, awards, pricing offer, or public SearchAction. The application has no public site-search endpoint. Its logo and social art are not presented as screenshots of an actual chat.

Schema.org properties were checked against [WebApplication](https://schema.org/WebApplication). Share fields follow [Open Graph](https://ogp.me/); canonical and static metadata follow [Google Search Central](https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls).

## Asset maintenance

`npm run generate:brand` regenerates SVG/PNG/ICO branding in both public asset directories using the repository's Playwright installation. Source: `scripts/generate-brand-assets.mjs`. `npm run test:metadata` validates the metadata contract and image dimensions.

The PNG social card is `public-hosted/social/drowse.png`; its editable vector source is beside it. The native dashboard has the same assets under `public/` and remains noindex.
