import { cpSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname } from "node:path";
import { describe, expect, test } from "bun:test";
import { MODERATION_LABELS, ToxicityClassifier } from "./index";
import { MODELS } from "./test-models";

describe("ToxicityClassifier", () => {
  test("is lazy: nothing loaded until the first score, explicit unload frees it, reload works", async () => {
    const moderator = new ToxicityClassifier({ modelsDir: MODELS });
    expect(moderator.isLoaded).toBe(false);
    const first = await moderator.score("hello");
    expect(moderator.isLoaded).toBe(true);
    await moderator.unload();
    expect(moderator.isLoaded).toBe(false);
    await moderator.unload(); // idempotent
    await moderator.load();
    expect(moderator.isLoaded).toBe(true);
    const again = await moderator.score("hello");
    expect(again.toxicity).toBeCloseTo(first.toxicity, 4);
    await moderator.unload();
  }, 60_000);

  test("each precision is its own self-contained folder", () => {
    const fp32 = new ToxicityClassifier({ modelsDir: MODELS });
    const q8 = new ToxicityClassifier({ precision: "q8", modelsDir: MODELS });
    expect(fp32.modelFolder).toBe("unbiased-toxic-roberta-detoxify-fp32");
    expect(q8.modelFolder).toBe("unbiased-toxic-roberta-detoxify-q8");
    expect(fp32.modelPath).toBe(`${fp32.modelDir}model.onnx`);
    expect(q8.modelPath.endsWith("/models/unbiased-toxic-roberta-detoxify-q8/model.onnx")).toBe(true);
    for (const dir of [fp32.modelDir, q8.modelDir]) {
      for (const file of ["model.onnx", "config.json", "tokenizer.json", "tokenizer_config.json", "README.md", "LICENSE"]) {
        expect(existsSync(`${dir}${file}`)).toBe(true);
      }
    }
  });

  test("loads a folder given as modelDir, whatever it is called", async () => {
    const dir = `${tmpdir()}/chat-moderation-byom-${process.pid}/mine`;
    cpSync(new ToxicityClassifier({ precision: "q8", modelsDir: MODELS }).modelDir, dir, { recursive: true });
    try {
      await using own = new ToxicityClassifier({ modelDir: dir });
      expect(own.modelDir).toBe(`${dir}/`);
      expect(own.modelPath).toBe(`${dir}/model.onnx`);
      const scores = await own.score("You are a useless idiot.");
      expect(scores.toxicity).toBeGreaterThan(0.99);
    } finally {
      rmSync(dirname(dir), { recursive: true, force: true });
    }
  }, 60_000);

  test("concurrent first calls share one load", async () => {
    await using moderator = new ToxicityClassifier({ precision: "q8", modelsDir: MODELS });
    const [a, b] = await Promise.all([moderator.score("hello"), moderator.score("hello")]);
    expect(a.toxicity).toBe(b.toxicity);
    expect(moderator.precision).toBe("q8");
  }, 60_000);

  test("scores an insult as toxic and a thank-you as clean", async () => {
    await using moderator = new ToxicityClassifier({ modelsDir: MODELS });
    const insult = await moderator.score("You are a useless idiot. Stop wasting my time.");
    expect(insult.toxicity).toBeGreaterThan(0.99);
    expect(insult.insult).toBeGreaterThan(0.99);
    const thanks = await moderator.score("thanks, that was really helpful!");
    expect(thanks.toxicity).toBeLessThan(0.01);
    for (const label of MODERATION_LABELS) expect(typeof insult[label]).toBe("number");
  }, 60_000);

  test("a toxic sentence past the 512-token window still counts", async () => {
    await using moderator = new ToxicityClassifier({ modelsDir: MODELS });
    const filler = "This paragraph is calm and describes a build log without any problem at all. ".repeat(45);
    const clean = await moderator.score(filler);
    const tail = await moderator.score(filler + "You are a useless idiot and I hope you choke on it.");
    expect(clean.toxicity).toBeLessThan(0.05);
    expect(tail.toxicity).toBeGreaterThan(clean.toxicity + 0.1);
  }, 60_000);
});
