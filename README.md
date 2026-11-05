# Chat Moderation

Toxicity and emotion scoring for Bun, local and CPU-only: Detoxify `unbiased` and GoEmotions, as ONNX, plus an `allcaps` ratio. Weights are in the repo.

```ts
import { Moderator } from "@tools/chat-moderation";

const moderator = new Moderator();
const { moderation, emotions, allcaps } = await moderator.score("some message");
moderation.toxicity;
emotions.anger;
allcaps;
await moderator.unload();
```

Tests: `bun test`. To try a message by hand, `echo 'some message' | bun cli.ts` prints every score, highest first (`--q8` for the quantized models); `cli.ts` is a dev tool, not part of the package.

## API

### `new Moderator(options?)`

Both models together. Construction is free and loads nothing. Each model loads on its first use and stays loaded until `unload()`.

- `score(text): Promise<Scores>`: `{ moderation, emotions, allcaps }` for one string. Loads whatever is not loaded yet.
- `scoreMany(texts): Promise<Scores[]>`: a list, in order.
- `load()`: loads both models up front. `unload()`: releases both. Both are no-ops for a model already in that state, and concurrent calls share one load.
- `isLoaded`: true when both models are loaded.
- `toxicityClassifier`, `emotionClassifier`: the two models as plain instances sharing the moderator's loaded state, for one model's scores alone.
- `precision`: `"fp32"` (default) or `"q8"`, applied to both models. See Models.
- `modelsDir`, `modelsUrl`: where model folders are read from and downloaded into, and where archives come from. See Models.
- `toxicityDir`, `emotionsDir`: bring your own model folders; each is passed to its classifier as `modelDir`.
- `[Symbol.asyncDispose]`: `await using moderator = new Moderator()` unloads at the end of the block.

### `new ToxicityClassifier(options?)`, `new EmotionClassifier(options?)`

One model each, same `precision`, `modelsDir` and `modelsUrl` options plus `modelDir` (bring your own: a folder holding `model.onnx`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, loaded as is; `precision` is then ignored and nothing is downloaded), same `score`, `scoreMany`, `load`, `unload`, `isLoaded`, and `await using`. Also `labels` (the model's labels, in output order), `modelFolder` (`<model>-<precision>`), `modelDir` (the absolute folder in use) and `modelPath` (absolute path of its `model.onnx`). Both extend `TextClassifier<Label>`, which holds the loading and long-text logic.

One call to a model returns every one of its labels: the classifier head produces all logits in a single forward pass and each is passed through a sigmoid. There is no way to ask for a subset, and none is needed.

### Types

- `ModerationScores`: `Record<ModerationLabel, number>`, each an independent sigmoid probability in `[0, 1]`. They do not sum to 1.
- `ModerationLabel`: the seven Detoxify scores, defined under Models; `MODERATION_LABELS` lists them in order.
- `EmotionScores`: `Record<EmotionLabel, number>`, likewise independent sigmoid probabilities.
- `EmotionLabel`: the 28 GoEmotions scores, defined under Models; `EMOTION_LABELS` lists them in order.
- `Scores`: `{ moderation: ModerationScores; emotions: EmotionScores; allcaps: number }`.
- `Precision`: `"fp32" | "q8"`.

### `allcaps(text): number`

The share of English letters (`A-Z`, `a-z`) in the text that are uppercase, as a float in `[0, 1]`: `"STOP"` is 1, `"stop"` is 0, `"Hello WORLD"` is 0.6. Digits, spaces, punctuation and non-ASCII letters are not counted on either side; text with no English letters is 0. No model is involved, so it is a plain function, also returned as `allcaps` by `Moderator.score`, computed on the whole text (not per chunk).

### Long text

Each model reads 512 tokens at a time. `score` splits longer text with `chunkByTokens`: whole sentences (`splitSentences`: on `.`, `!`, `?`, newlines) while they fit, then words, then raw token slices for a single oversized run such as a URL. Every chunk is scored and the maximum per label is returned, so a toxic or furious sentence at the end of a long message counts. Text that fits is scored whole. Both helpers are exported.

## Models

Four models, one folder each, named `<model>-<precision>`: `unbiased-toxic-roberta-detoxify-fp32`, `unbiased-toxic-roberta-detoxify-q8`, `roberta-base-go_emotions-fp32`, `roberta-base-go_emotions-q8`. A folder is self-contained: `model.onnx` next to the `config.json`, `tokenizer.json` and `tokenizer_config.json` Transformers.js reads, the slow-tokenizer sources (`vocab.json`, `merges.txt`, `special_tokens_map.json`), a `README.md` on provenance, and the model's `LICENSE`. fp32 is the exact export of the checkpoint. q8 is the same weights rounded to int8: a quarter of the file size, under a third of the memory, and scores that differ from fp32 by the amount given under each model, so it is a memory trade, not a free one. With both models loaded the process takes about 1.5 GB at fp32 and about 435 MB at q8.

### Where they are read from

The package ships no weights. A classifier looks for its folder under one directory, and reads nothing anywhere else:

|         | directory                                                                          |
| ------- | ---------------------------------------------------------------------------------- |
| option  | `modelsDir`                                                                        |
| macOS   | `~/Library/Caches/chat-moderation/models/`                                         |
| Linux   | `$XDG_CACHE_HOME/chat-moderation/models/`, else `~/.cache/chat-moderation/models/` |
| Windows | `%LOCALAPPDATA%\chat-moderation\models\`                                           |

First match wins, top to bottom. `modelsDir()` returns it. A `modelDir` passed directly bypasses all of this.

### Where they come from

If `<modelsDir>/<model>-<precision>/model.onnx` is missing, the first `load()` or `score()` downloads `<model>-<precision>.7z` from a base URL (option `modelsUrl`, else `baseUrl` in `src/models.json`), checks its size and sha256 against `src/models.json`, extracts it into `modelsDir` with the bundled 7-Zip, and loads it. Only the precision in use is fetched: q8 for both models is 56 MB plus 97 MB; fp32 is 466 MB plus 430 MB. A failed check deletes the download and throws. Two processes starting at once take turns on a lock folder under `<modelsDir>/.tmp/`. `ensureModel(name, options)` is exported to do this ahead of time.

The archives are GitHub release assets, one per folder, named exactly `<model>-<precision>.7z`. `baseUrl` in `src/models.json` is empty until that release exists; until then pass `modelsUrl`, or put the folders in place yourself. Tests and `cli.ts` in this checkout pass `modelsDir` pointing at `models/` here.

### Detoxify `unbiased`

A RoBERTa-base classifier trained by Unitary on the Jigsaw Unintended Bias in Toxicity dataset, checkpoint `toxic_debiased-c7548aa0.ckpt` from the Detoxify GitHub release, exported to ONNX under `models/unbiased-toxic-roberta-detoxify/`. The seven scores in `moderation`, with Jigsaw's definitions: the first two from the rater schema Detoxify's README quotes, the rest from the Perspective API attribute descriptions.

| score             | what it measures                                                                                                                               |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `toxicity`        | a rude, disrespectful, or unreasonable comment that is somewhat likely to make you leave a discussion or give up on sharing your perspective   |
| `severe_toxicity` | a very hateful, aggressive, or disrespectful comment that is very likely to make you leave a discussion or give up on sharing your perspective |
| `obscene`         | obscene or vulgar language such as cursing                                                                                                     |
| `identity_attack` | negative or hateful comments targeting someone because of their identity                                                                       |
| `insult`          | insulting, inflammatory, or negative comment towards a person or a group of people                                                             |
| `threat`          | describes an intention to inflict pain, injury, or violence against an individual or group                                                     |
| `sexual_explicit` | contains references to sexual acts, body parts, or other lewd content                                                                          |

Loaded, fp32 takes about 860 MB of process memory and q8 about 255 MB. q8 scores differ from fp32 by up to about 0.04.

The hub's `unitary/unbiased-toxic-roberta` (and the `onnx-community` conversion of it) is an older checkpoint that scores shouty text very differently (0.40 vs 0.99 toxicity on an all-caps rant), which is why this export exists.

### GoEmotions

`SamLowe/roberta-base-go_emotions`: RoBERTa-base fine-tuned on Google's GoEmotions (Reddit comments, 28 labels, multi-label). The Transformers.js ONNX export of it, `MicahB/roberta-base-go_emotions`, is committed under `models/roberta-base-go_emotions/`, verified by sha256 against the hub. The 28 scores in `emotions`, in the model's order, with the definitions from Appendix A of the GoEmotions paper (Demszky et al., 2020):

| score            | what it measures                                                                                                        |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `admiration`     | finding something impressive or worthy of respect                                                                       |
| `amusement`      | finding something funny or being entertained                                                                            |
| `anger`          | a strong feeling of displeasure or antagonism                                                                           |
| `annoyance`      | mild anger, irritation                                                                                                  |
| `approval`       | having or expressing a favorable opinion                                                                                |
| `caring`         | displaying kindness and concern for others                                                                              |
| `confusion`      | lack of understanding, uncertainty                                                                                      |
| `curiosity`      | a strong desire to know or learn something                                                                              |
| `desire`         | a strong feeling of wanting something or wishing for something to happen                                                |
| `disappointment` | sadness or displeasure caused by the nonfulfillment of one's hopes or expectations                                      |
| `disapproval`    | having or expressing an unfavorable opinion                                                                             |
| `disgust`        | revulsion or strong disapproval aroused by something unpleasant or offensive                                            |
| `embarrassment`  | self-consciousness, shame, or awkwardness                                                                               |
| `excitement`     | feeling of great enthusiasm and eagerness                                                                               |
| `fear`           | being afraid or worried                                                                                                 |
| `gratitude`      | a feeling of thankfulness and appreciation                                                                              |
| `grief`          | intense sorrow, especially caused by someone's death                                                                    |
| `joy`            | a feeling of pleasure and happiness                                                                                     |
| `love`           | a strong positive emotion of regard and affection                                                                       |
| `nervousness`    | apprehension, worry, anxiety                                                                                            |
| `optimism`       | hopefulness and confidence about the future or the success of something                                                 |
| `pride`          | pleasure or satisfaction due to one's own achievements or the achievements of those with whom one is closely associated |
| `realization`    | becoming aware of something                                                                                             |
| `relief`         | reassurance and relaxation following release from anxiety or distress                                                   |
| `remorse`        | regret or guilty feeling                                                                                                |
| `sadness`        | emotional pain, sorrow                                                                                                  |
| `surprise`       | feeling astonished, startled by something unexpected                                                                    |
| `neutral`        | none of the 27 emotions; the paper keeps it outside the emotion set                                                     |

Loaded, fp32 takes about 920 MB of process memory and q8 about 270 MB. q8 scores differ from fp32 by up to about 0.14, enough to move a label across a 0.5 threshold.

# License

This software is available under either the [MIT License](LICENSE)
or the [BSD Zero Clause License (0BSD)](LICENSE-0BSD),
at your option.
