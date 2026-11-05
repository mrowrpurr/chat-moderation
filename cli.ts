// Dev CLI, not part of the package: score text from stdin, print every score
// on its own line, highest first.
//
//     echo 'some message' | bun cli.ts
//     bun cli.ts --q8 < file.txt
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { Moderator } from "./src/index";

const checkoutModels = fileURLToPath(new URL("./models/", import.meta.url));

const precision = process.argv.includes("--q8") ? "q8" : "fp32";
const text = (await Bun.stdin.text()).trim();
if (!text) {
  console.error("Usage: echo 'some message' | bun cli.ts [--q8]");
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
