// The moderator: toxicity and emotions on every text, in one call. 🍞
import { allcaps } from "./allcaps.ts";
import { EmotionClassifier, type EmotionScores } from "./emotions.ts";
import { ToxicityClassifier, type ModerationScores } from "./toxicity.ts";
import type { ClassifierOptions } from "./classifier.ts";

export type Scores = {
  moderation: ModerationScores;
  emotions: EmotionScores;
  /** Share of English letters that are uppercase, in [0, 1]. See `allcaps`. */
  allcaps: number;
};

export type ModeratorOptions = Omit<ClassifierOptions, "modelDir"> & {
  /** Bring your own Detoxify folder; see `ClassifierOptions.modelDir`. */
  toxicityDir?: string;
  /** Bring your own GoEmotions folder; see `ClassifierOptions.modelDir`. */
  emotionsDir?: string;
};

/**
 * A `ToxicityClassifier` and an `EmotionClassifier` driven together.
 *
 *     const moderator = new Moderator();            // nothing loaded yet
 *     const { moderation, emotions, allcaps } = await moderator.score(text);
 *     moderation.toxicity; emotions.anger; allcaps;
 *     await moderator.unload();                      // frees both models
 *
 * Each model loads lazily on first use and stays loaded until `unload()`.
 * `load()` loads both up front. For one model's scores alone, use
 * `moderator.toxicityClassifier` or `moderator.emotionClassifier` directly; they
 * are plain instances and share their loaded state with the moderator.
 */
export class Moderator {
  readonly toxicityClassifier: ToxicityClassifier;
  readonly emotionClassifier: EmotionClassifier;

  constructor(options: ModeratorOptions = {}) {
    const { toxicityDir, emotionsDir, ...shared } = options;
    this.toxicityClassifier = new ToxicityClassifier({ ...shared, modelDir: toxicityDir });
    this.emotionClassifier = new EmotionClassifier({ ...shared, modelDir: emotionsDir });
  }

  get precision() {
    return this.toxicityClassifier.precision;
  }

  /** True when both models are loaded. */
  get isLoaded(): boolean {
    return this.toxicityClassifier.isLoaded && this.emotionClassifier.isLoaded;
  }

  /** Loads both models. No-op for any already loaded. */
  async load(): Promise<void> {
    await Promise.all([this.toxicityClassifier.load(), this.emotionClassifier.load()]);
  }

  /** Releases both models. No-op for any not loaded. */
  async unload(): Promise<void> {
    await Promise.all([this.toxicityClassifier.unload(), this.emotionClassifier.unload()]);
  }

  /** Both models' scores plus `allcaps` for one text. Loads whatever is not loaded yet. */
  async score(text: string): Promise<Scores> {
    const [moderation, emotions] = await Promise.all([
      this.toxicityClassifier.score(text),
      this.emotionClassifier.score(text),
    ]);
    return { moderation, emotions, allcaps: allcaps(text) };
  }

  /** Both models' scores for each text, in order. */
  async scoreMany(texts: readonly string[]): Promise<Scores[]> {
    const out: Scores[] = [];
    for (const text of texts) out.push(await this.score(text));
    return out;
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.unload();
  }
}
