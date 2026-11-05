// Detoxify `unbiased` toxicity scoring. 🍞
import { TextClassifier, type ClassifierOptions } from "./classifier.ts";

/** The seven Detoxify labels, in the order Detoxify reports them. */
export const MODERATION_LABELS = [
  "toxicity",
  "severe_toxicity",
  "obscene",
  "identity_attack",
  "insult",
  "threat",
  "sexual_explicit",
] as const;

export type ModerationLabel = (typeof MODERATION_LABELS)[number];

/** Each label's independent sigmoid probability in [0, 1]. They do not sum to 1. */
export type ModerationScores = Record<ModerationLabel, number>;

export type ToxicityClassifierOptions = ClassifierOptions;

/**
 * Detoxify's `unbiased` model: the real `toxic_debiased-c7548aa0.ckpt` from
 * Detoxify's GitHub release, exported to ONNX and committed under
 * models/unbiased-toxic-roberta-detoxify-{fp32,q8}/.
 * The hub's unitary/unbiased-toxic-roberta (and onnx-community's conversion of
 * it) is an older checkpoint that scores shouty text very differently, so we
 * don't use it.
 *
 *     const toxicity = new ToxicityClassifier();   // nothing loaded yet
 *     const scores = await toxicity.score(text);     // loads on first use
 *     await toxicity.unload();                       // frees the model
 *
 * See `TextClassifier` for loading and long-text behaviour.
 */
export class ToxicityClassifier extends TextClassifier<ModerationLabel> {
  constructor(options: ToxicityClassifierOptions = {}) {
    super("unbiased-toxic-roberta-detoxify", MODERATION_LABELS, options);
  }
}
