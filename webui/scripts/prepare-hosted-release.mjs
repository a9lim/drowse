import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { releaseRobots } from "./agent-discovery.mjs";

const root = resolve("dist-hosted");
const headersPath = resolve(root, "_headers");
const [headers, index, robots, builtLicense, sourceLicense] = await Promise.all([
  readFile(headersPath, "utf8"),
  readFile(resolve(root, "index.html"), "utf8"),
  readFile(resolve(root, "robots.txt"), "utf8"),
  readFile(resolve(root, "LICENSE"), "utf8"),
  readFile(resolve("..", "LICENSE"), "utf8"),
]);
const metadata = (name) => {
  const match = new RegExp(`<meta name="${name}" content="([^"]+)" \\/>`).exec(index);
  if (!match) throw new Error(`hosted release index is missing ${name}`);
  return match[1];
};
const revision = metadata("drowse-source-revision");
const canonical = /<link rel="canonical" href="(https:\/\/[^"\s]+\/)"/.exec(index)?.[1];
if (!canonical) throw new Error("Set DROWSE_PUBLIC_ORIGIN to the public HTTPS origin before building a hosted release.");
if (metadata("drowse-release-channel") !== "release") {
  throw new Error("hosted release output was not compiled in release mode");
}
if (!/^[0-9a-f]{40}$/.test(revision)) {
  throw new Error("hosted release output has no exact source revision");
}
if (metadata("drowse-source-url") !== `https://github.com/a9lim/drowse/tree/${revision}`) {
  throw new Error("hosted release source URL does not match its revision");
}
if (builtLicense !== sourceLicense) {
  throw new Error("hosted release does not contain the repository AGPL license");
}
if (robots !== "User-agent: *\nDisallow: /\n") {
  throw new Error("hosted release must be promoted from a noindex preview build");
}
const publicHeaders = headers.replace(/^\s*X-Robots-Tag: noindex, nofollow\s*$/m, "");
if (publicHeaders === headers) {
  throw new Error("preview noindex header was missing from the hosted build");
}
if (/X-Robots-Tag:\s*noindex/i.test(publicHeaders.split(/\n\s*\n/)[0])) {
  throw new Error("hosted release output still contains a global noindex header");
}
await writeFile(headersPath, `${publicHeaders}\n/app\n  X-Robots-Tag: noindex, follow\n\n/app/*\n  X-Robots-Tag: noindex, follow\n`);
await writeFile(resolve(root, "robots.txt"), releaseRobots(canonical));
