# Font provenance

## Wix Madefor Display and Wix Madefor Text

- Source project: `wix-incubator/wixmadefor`, distributed under SIL OFL 1.1.
- Browser files: official Google Fonts WOFF2 variable subsets fetched on
  2026-08-27 from `fonts.gstatic.com`; Display URL version `v12`, Text URL
  version `v17`.
- Axes: Display and Text expose `wght` 400–800. Text includes normal and
  italic files. Subsets cover Latin, extended Latin, Vietnamese, Cyrillic,
  and extended Cyrillic.
- License: `LICENSE-Wix-Madefor.txt` here and in both public directories.

## Martian Mono

- Source project: `evilmartians/mono`, distributed under SIL OFL 1.1.
- Browser files: official Google Fonts WOFF2 variable subsets fetched on
  2026-08-27 from `fonts.gstatic.com`, URL version `v6`.
- Axes: `wght` 100–800 and `wdth` 75%–112.5%. Subsets cover Latin, extended
  Latin, Cyrillic, and extended Cyrillic.
- License: `LICENSE-Martian-Mono.txt` here and in both public directories.

Every local font file is pinned by `SHA256SUMS`. The app consumes the Wix
families and Martian Mono through `src/lib/style/fonts.css` with no runtime
font request to a third-party origin.
