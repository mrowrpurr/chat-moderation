import { existsSync } from "node:fs";
import { describe, expect, test } from "./testing.ts";
import { EMOTION_LABELS, EmotionClassifier } from "./index.ts";
import { MODELS } from "./test-models.ts";

describe("EmotionClassifier", () => {
  test("is lazy, unloads, reloads", async () => {
    const emotions = new EmotionClassifier({ modelsDir: MODELS });
    expect(emotions.isLoaded).toBe(false);
    const first = await emotions.score("hello");
    expect(emotions.isLoaded).toBe(true);
    await emotions.unload();
    expect(emotions.isLoaded).toBe(false);
    await emotions.load();
    const again = await emotions.score("hello");
    expect(again.neutral).toBeCloseTo(first.neutral, 4);
    await emotions.unload();
  }, 60_000);

  test("each precision is its own self-contained folder", () => {
    const fp32 = new EmotionClassifier({ modelsDir: MODELS });
    const q8 = new EmotionClassifier({ precision: "q8", modelsDir: MODELS });
    expect(fp32.modelFolder).toBe("roberta-base-go_emotions-fp32");
    expect(q8.modelFolder).toBe("roberta-base-go_emotions-q8");
    expect(fp32.modelPath).toBe(`${fp32.modelDir}model.onnx`);
    expect(q8.modelPath.endsWith("/models/roberta-base-go_emotions-q8/model.onnx")).toBe(true);
    for (const dir of [fp32.modelDir, q8.modelDir]) {
      for (const file of ["model.onnx", "config.json", "tokenizer.json", "tokenizer_config.json", "README.md", "LICENSE"]) {
        expect(existsSync(`${dir}${file}`)).toBe(true);
      }
    }
  });

  test("concurrent first calls share one load", async () => {
    await using emotions = new EmotionClassifier({ modelsDir: MODELS });
    const [a, b] = await Promise.all([emotions.score("hello"), emotions.score("hello")]);
    expect(a.neutral).toBe(b.neutral);
    expect(emotions.isLoaded).toBe(true);
  }, 60_000);

  test("reports all 28 labels as independent probabilities", async () => {
    await using emotions = new EmotionClassifier({ precision: "q8", modelsDir: MODELS });
    const scores = await emotions.score("I can't believe they cancelled the show, I'm gutted.");
    expect(Object.keys(scores).sort()).toEqual([...EMOTION_LABELS].sort());
    let sum = 0;
    for (const label of EMOTION_LABELS) {
      expect(scores[label]).toBeGreaterThanOrEqual(0);
      expect(scores[label]).toBeLessThanOrEqual(1);
      sum += scores[label];
    }
    expect(sum).not.toBeCloseTo(1, 1); // sigmoid heads, not a softmax
  }, 60_000);

  test("hears anger in a rant and gratitude in a thank-you", async () => {
    await using emotions = new EmotionClassifier({ modelsDir: MODELS });
    const rant = await emotions.score("I am so angry right now, this is absolutely infuriating!");
    expect(rant.anger).toBeGreaterThan(0.5);
    expect(rant.gratitude).toBeLessThan(0.05);
    const thanks = await emotions.score("Thank you so much, that was really helpful!");
    expect(thanks.gratitude).toBeGreaterThan(0.9);
    expect(thanks.anger).toBeLessThan(0.05);
  }, 60_000);

  test("an angry sentence past the 512-token window still counts", async () => {
    await using emotions = new EmotionClassifier({ modelsDir: MODELS });
    const filler = "This paragraph is calm and describes a build log without any problem at all. ".repeat(45);
    const calm = await emotions.score(filler);
    const tail = await emotions.score(filler + "I am so angry right now, this is absolutely infuriating!");
    expect(tail.anger).toBeGreaterThan(calm.anger + 0.3);
  }, 60_000);
});
