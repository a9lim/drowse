import { parseArgs } from "node:util";
import { writeFile } from "node:fs/promises";
import puppeteer from "puppeteer";
import { runNativeEvidence } from "./webmcp-live-scenario.mjs";

const { values } = parseArgs({ options: { url: { type: "string" }, output: { type: "string" }, control: { type: "string" }, disconnect: { type: "boolean", default: true } } });
if (!values.url || !values.output) throw Error("Provide --url for an isolated real Python dashboard and --output for the evidence JSON");
const browser = await puppeteer.launch({ headless: true, args: ["--enable-features=WebMCP"] });
try {
  const page = await browser.newPage();
  await page.goto(values.url);
  const result = await runNativeEvidence(page, { runtime: "http", control: values.control, disconnect: values.disconnect });
  await writeFile(values.output, JSON.stringify({ passed: true, releaseEvidence: false, ...result }, null, 2) + "\n", { flag: "wx" });
  console.log(`Native WebMCP real HTTP evidence written to ${values.output}`);
} catch (error) {
  await writeFile(values.output, JSON.stringify({ passed: false, releaseEvidence: false, error: error.stack }, null, 2) + "\n", { flag: "wx" });
  throw error;
} finally { await browser.close(); }
