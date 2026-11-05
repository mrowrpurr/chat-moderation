// Dev CLI, not part of the package: score text from stdin, print every score
// on its own line, highest first.
//
//     echo 'some message' | bun cli.ts
//     bun cli.ts --q8 < file.txt
//     node cli.ts --q8 < file.txt
import { existsSync } from "node:fs";
import { text as readText } from "node:stream/consumers";
import { fileURLToPath } from "node:url";
import { Moderator } from "./src/index.ts";

const checkoutModels = fileURLToPath(new URL("./models/", import.meta.url));

const precision = process.argv.includes("--q8") ? "q8" : "fp32";
const text = (await readText(process.stdin)).trim();
if (!text) {
  console.error("Usage: echo 'some message' | bun cli.ts [--q8]   (or node cli.ts)");
  process.exit(2);
}

await using moderator = new Moderator({ precision, modelsDir: existsSync(checkoutModels) ? checkoutModels : undefined });
const { moderation, emotions, allcaps } = await moderator.score(text);

const rows = [
  { label: "allcaps", score: allcaps },
  ...Object.entries(moderation).map(([label, score]) => ({ label: `moderation:${label}`, score })),
  ...Object.entries(emotions).map(([label, score]) => ({ label: `emotions:${label}`, score })),
].sort((a, b) => b.score - a.score);

for (const { label, score } of rows) console.log(`${score.toFixed(4)}  ${label}`);
