// Split text that exceeds the model's token window into chunks the model can
// score whole: on sentence boundaries where possible, then word boundaries,
// then raw token slices for a single oversized "word" (a URL, base64, ...). 🍞

export type Tokenizer = {
  /** Token ids for `text`, without special tokens. */
  encode(text: string): number[];
  /** Text for `ids`, without special tokens. */
  decode(ids: number[]): string;
};

/** Sentences and lines, trimmed, empties dropped. */
export function splitSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+|\n+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/** Pieces that each fit in `maxTokens`: whole sentences, else words, else token slices. */
function* piecesWithinBudget(text: string, tokenizer: Tokenizer, maxTokens: number): Generator<string> {
  for (const sentence of splitSentences(text)) {
    if (tokenizer.encode(sentence).length <= maxTokens) {
      yield sentence;
      continue;
    }
    for (const word of sentence.split(/\s+/)) {
      const ids = tokenizer.encode(word);
      if (ids.length <= maxTokens) {
        yield word;
        continue;
      }
      for (let i = 0; i < ids.length; i += maxTokens) {
        yield tokenizer.decode(ids.slice(i, i + maxTokens));
      }
    }
  }
}

/**
 * Pack `text` into chunks of at most `maxTokens` tokens each. Text that already
 * fits comes back as a single chunk, untouched.
 */
export function chunkByTokens(text: string, tokenizer: Tokenizer, maxTokens: number): string[] {
  if (tokenizer.encode(text).length <= maxTokens) return [text];

  const chunks: string[] = [];
  let current: string[] = [];
  let currentTokens = 0;
  const flush = () => {
    if (current.length) chunks.push(current.join(" "));
    current = [];
    currentTokens = 0;
  };

  for (const piece of piecesWithinBudget(text, tokenizer, maxTokens)) {
    const n = tokenizer.encode(piece).length;
    if (currentTokens + n > maxTokens) flush();
    current.push(piece);
    currentTokens += n;
  }
  flush();
  return chunks;
}
