import { describe, expect, test } from "./testing.ts";
import { allcaps } from "./allcaps.ts";

describe("allcaps", () => {
  test("all uppercase is 1", () => {
    expect(allcaps("STOP YELLING")).toBe(1);
  });

  test("all lowercase is 0", () => {
    expect(allcaps("stop yelling")).toBe(0);
  });

  test("mixed case is the uppercase share of the letters", () => {
    expect(allcaps("Hello WORLD")).toBeCloseTo(6 / 10, 10);
  });

  test("digits, spaces and punctuation are not letters", () => {
    expect(allcaps("ABC 123 !!! ... abc")).toBeCloseTo(0.5, 10);
    expect(allcaps("ABC 123")).toBe(1);
  });

  test("non-ASCII letters are not counted on either side", () => {
    expect(allcaps("ÉÉÉÉ a")).toBe(0);
    expect(allcaps("éééé A")).toBe(1);
  });

  test("no English letters is 0", () => {
    expect(allcaps("")).toBe(0);
    expect(allcaps("123 ... 🍞")).toBe(0);
  });
});
