import { describe, expect, test } from "bun:test";
import { EMOTION_LABELS, MODERATION_LABELS, Moderator } from "./index";
import { MODELS } from "./test-models";

describe("Moderator", () => {
  test("toxicityDir and emotionsDir reach the right classifier", () => {
    const m = new Moderator({ toxicityDir: "/a/tox", emotionsDir: "/b/emo" });
    expect(m.toxicityClassifier.modelDir).toBe("/a/tox/");
    expect(m.emotionClassifier.modelDir).toBe("/b/emo/");
  });

  test("scores with both models and unloads both", async () => {
    const moderator = new Moderator({ modelsDir: MODELS });
    expect(moderator.isLoaded).toBe(false);
    const { moderation, emotions, allcaps } = await moderator.score("You are a useless idiot. Stop wasting my time.");
    expect(moderator.isLoaded).toBe(true);
    expect(moderator.toxicityClassifier.isLoaded).toBe(true);
    expect(moderator.emotionClassifier.isLoaded).toBe(true);
    expect(moderation.toxicity).toBeGreaterThan(0.99);
    expect(allcaps).toBeCloseTo(2 / 36, 10);
    expect(emotions.anger + emotions.annoyance + emotions.disapproval).toBeGreaterThan(0.5);
    for (const label of MODERATION_LABELS) expect(typeof moderation[label]).toBe("number");
    for (const label of EMOTION_LABELS) expect(typeof emotions[label]).toBe("number");
    await moderator.unload();
    expect(moderator.toxicityClassifier.isLoaded).toBe(false);
    expect(moderator.emotionClassifier.isLoaded).toBe(false);
  }, 90_000);

  test("load() loads both up front; one model can be used alone", async () => {
    await using moderator = new Moderator({ precision: "q8", modelsDir: MODELS });
    await moderator.load();
    expect(moderator.isLoaded).toBe(true);
    expect(moderator.precision).toBe("q8");
    const alone = await moderator.emotionClassifier.score("Thank you so much!");
    expect(alone.gratitude).toBeGreaterThan(0.9);
    const both = await moderator.scoreMany(["Thank you so much!", "I hate this."]);
    expect(both).toHaveLength(2);
    expect(both[0]!.emotions.gratitude).toBeCloseTo(alone.gratitude, 6);
    expect(both[1]!.moderation.toxicity).toBeGreaterThan(both[0]!.moderation.toxicity);
  }, 90_000);
});
