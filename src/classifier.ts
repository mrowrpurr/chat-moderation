// One ONNX text classifier run locally in Bun via Transformers.js: lazy load,
// explicit unload, long text chunked and scored as the max per label. 🍞
import { existsSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { env, pipeline, type TextClassificationPipeline } from "@huggingface/transformers";
import { chunkByTokens, type Tokenizer } from "./chunk";
import { ensureModel, modelsDir } from "./models";

/**
 * Which ONNX weights to run. `fp32` is the exact export of the checkpoint.
 * `q8` has dynamically quantized int8 weights: smaller, faster, scores drift
 * by up to a few hundredths from fp32. See the README for measurements.
 */
export type Precision = "fp32" | "q8";

export type ClassifierOptions = {
  /** Default `"fp32"`. Ignored when `modelDir` is given. */
  precision?: Precision;
  /**
   * Bring your own model: a folder holding `model.onnx`, `config.json`,
   * `tokenizer.json` and `tokenizer_config.json`. Loaded as is.
   */
  modelDir?: string;
  /** Where `<model>-<precision>/` folders are read from and downloaded into. Default: see `modelsDir()`. */
  modelsDir?: string;
  /** Base URL the archives are downloaded from. Default: see `modelsUrl()`. */
  modelsUrl?: string;
};

const MODEL_FILE = "model.onnx";

type Score = { label: string; score: number };

type Loaded = {
  pipe: TextClassificationPipeline;
  tokenizer: Tokenizer;
  /** Tokens of content per chunk: the model's window minus its two special tokens. */
  chunkBudget: number;
};

/**
 * Shared machinery for `ToxicityClassifier` and `EmotionClassifier`. Construction is
 * free. The model is loaded lazily on the first `score`, or explicitly with
 * `load()`. `unload()` releases it; the instance can be loaded again
 * afterwards. `await using` unloads at the end of the block.
 *
 * Text longer than the model's token window is split on sentence, then word,
 * then token boundaries; every chunk is scored and the maximum per label is
 * reported, so a strong sentence at the end of a long message is not lost.
 * Text that fits is scored whole.
 */
export abstract class TextClassifier<Label extends string> {
  readonly precision: Precision;
  private readonly ownDir: string | null;
  private readonly baseDir: string;
  private readonly modelsUrl: string | undefined;
  private loaded: Loaded | null = null;
  private loading: Promise<Loaded> | null = null;

  protected constructor(
    /** Model name; the folder under `models/` is `<modelName>-<precision>`. */
    readonly modelName: string,
    /** Every label the model reports, in its output order. */
    readonly labels: readonly Label[],
    options: ClassifierOptions = {},
  ) {
    this.precision = options.precision ?? "fp32";
    this.ownDir = options.modelDir ? resolve(options.modelDir) + "/" : null;
    this.baseDir = options.modelsDir ? resolve(options.modelsDir) : modelsDir();
    this.modelsUrl = options.modelsUrl;
  }

  /** Folder name under `models/` for this model at this precision. */
  get modelFolder(): string {
    return `${this.modelName}-${this.precision}`;
  }

  /** Absolute path of the folder this instance loads from: `modelDir` if given, else `<modelsDir>/<modelFolder>/`. */
  get modelDir(): string {
    return this.ownDir ?? join(this.baseDir, this.modelFolder) + "/";
  }

  /** Absolute path of the ONNX file this instance runs. */
  get modelPath(): string {
    return `${this.modelDir}${MODEL_FILE}`;
  }

  get isLoaded(): boolean {
    return this.loaded !== null;
  }

  /** Loads the model. No-op if already loaded; concurrent calls share one load. */
  async load(): Promise<void> {
    await this.ensureLoaded();
  }

  /** Releases the model. No-op if not loaded. `load()` or `score()` can bring it back. */
  async unload(): Promise<void> {
    if (this.loading) await this.loading.catch(() => {});
    const loaded = this.loaded;
    this.loaded = null;
    if (loaded) await loaded.pipe.dispose();
  }

  /** Every label's sigmoid probability for one text. Loads the model first if needed. */
  async score(text: string): Promise<Record<Label, number>> {
    const { pipe, tokenizer, chunkBudget } = await this.ensureLoaded();
    const result = Object.fromEntries(this.labels.map((l) => [l, 0])) as Record<Label, number>;
    for (const chunk of chunkByTokens(text, tokenizer, chunkBudget)) {
      const scores = (await pipe(chunk, { top_k: null })) as Score[];
      for (const { label, score } of scores) {
        const l = label as Label;
        if (l in result && score > result[l]) result[l] = score;
      }
    }
    return result;
  }

  /** Scores for each text, in order. */
  async scoreMany(texts: readonly string[]): Promise<Record<Label, number>[]> {
    const out: Record<Label, number>[] = [];
    for (const text of texts) out.push(await this.score(text));
    return out;
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.unload();
  }

  private ensureLoaded(): Promise<Loaded> {
    if (this.loaded) return Promise.resolve(this.loaded);
    if (!this.loading) {
      this.loading = this.loadModel().then(
        (loaded) => {
          this.loaded = loaded;
          this.loading = null;
          return loaded;
        },
        (err) => {
          this.loading = null;
          throw err;
        },
      );
    }
    return this.loading;
  }

  private async loadModel(): Promise<Loaded> {
    if (!this.ownDir) {
      await ensureModel(this.modelFolder, { modelsDir: this.baseDir, modelsUrl: this.modelsUrl });
    }
    if (!existsSync(this.modelPath)) {
      throw new Error(`ONNX model not found at ${this.modelPath}.`);
    }
    env.localModelPath = dirname(this.modelDir) + "/";
    env.allowLocalModels = true;
    env.useFSCache = false; // read our folder only; never Transformers.js's own cache
    // Each folder holds exactly one ONNX file, so the precision is in the folder
    // name, not in Transformers.js's dtype suffix: dtype stays "fp32" (suffix "")
    // and subfolder "" replaces its default "onnx/".
    const pipe = await pipeline("text-classification", basename(this.modelDir), {
      device: "cpu",
      dtype: "fp32",
      subfolder: "",
    });
    const t = pipe.tokenizer;
    return {
      pipe,
      tokenizer: {
        encode: (text) => t.encode(text, { add_special_tokens: false }),
        decode: (ids) => t.decode(ids, { skip_special_tokens: true }),
      },
      chunkBudget: t.model_max_length - 2,
    };
  }
}
