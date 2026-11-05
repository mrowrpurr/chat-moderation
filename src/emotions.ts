// GoEmotions emotion scoring with SamLowe's roberta-base-go_emotions. 🍞
import { TextClassifier, type ClassifierOptions } from "./classifier";

/** The 28 GoEmotions labels, in the model's output order. */
export const EMOTION_LABELS = [
  "admiration",
  "amusement",
  "anger",
  "annoyance",
  "approval",
  "caring",
  "confusion",
  "curiosity",
  "desire",
  "disappointment",
  "disapproval",
  "disgust",
  "embarrassment",
  "excitement",
  "fear",
  "gratitude",
  "grief",
  "joy",
  "love",
  "nervousness",
  "optimism",
  "pride",
  "realization",
  "relief",
  "remorse",
  "sadness",
  "surprise",
  "neutral",
] as const;

export type EmotionLabel = (typeof EMOTION_LABELS)[number];

/** Each label's independent sigmoid probability in [0, 1]. They do not sum to 1. */
export type EmotionScores = Record<EmotionLabel, number>;

export type EmotionClassifierOptions = ClassifierOptions;

/**
 * SamLowe/roberta-base-go_emotions (roberta-base fine-tuned on Google's
 * GoEmotions, multi-label), the ONNX export published as
 * MicahB/roberta-base-go_emotions, committed under
 * models/roberta-base-go_emotions-{fp32,q8}/.
 *
 *     const emotions = new EmotionClassifier();     // nothing loaded yet
 *     const scores = await emotions.score(text);     // loads on first use
 *     scores.anger;
 *
 * See `TextClassifier` for loading and long-text behaviour.
 */
export class EmotionClassifier extends TextClassifier<EmotionLabel> {
  constructor(options: EmotionClassifierOptions = {}) {
    super("roberta-base-go_emotions", EMOTION_LABELS, options);
  }
}
