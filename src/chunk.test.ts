import { describe, expect, test } from "bun:test";
import { chunkByTokens, splitSentences, type Tokenizer } from "./chunk";

// One token per whitespace-separated word; decode joins them back.
const words: Tokenizer = {
  encode: (t) => t.split(/\s+/).filter(Boolean).map((_, i) => i),
  decode: (ids) => ids.map(() => "x").join(" "),
};

describe("splitSentences", () => {
  test("splits on sentence punctuation and newlines, drops empties", () => {
    expect(splitSentences("One. Two!  Three?\n\nFour\n")).toEqual(["One.", "Two!", "Three?", "Four"]);
  });
});

describe("chunkByTokens", () => {
  test("text within budget is one untouched chunk", () => {
    expect(chunkByTokens("a b c.\nd e", words, 5)).toEqual(["a b c.\nd e"]);
  });

  test("packs whole sentences up to the budget", () => {
    const text = "one two three. four five. six seven eight. nine.";
    expect(chunkByTokens(text, words, 5)).toEqual(["one two three. four five.", "six seven eight. nine."]);
  });

  test("a sentence over budget falls back to word boundaries", () => {
    const text = "a b c d e f g.";
    expect(chunkByTokens(text, words, 3)).toEqual(["a b c", "d e f", "g."]);
  });

  test("a single oversized word is sliced by tokens", () => {
    const chars: Tokenizer = {
      encode: (t) => [...t.replace(/\s+/g, "")].map((_, i) => i),
      decode: (ids) => "x".repeat(ids.length),
    };
    expect(chunkByTokens("abcdefg", chars, 3)).toEqual(["xxx", "xxx", "x"]);
  });

  test("every chunk fits the budget", () => {
    const text = Array.from({ length: 50 }, (_, i) => `sentence number ${i} has some words in it.`).join(" ");
    for (const chunk of chunkByTokens(text, words, 20)) {
      expect(words.encode(chunk).length).toBeLessThanOrEqual(20);
    }
  });
});
