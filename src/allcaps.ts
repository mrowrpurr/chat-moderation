// allcaps: how much of the text is shouted. 🍞
//
// The share of English letters (A-Z, a-z) that are uppercase, as a float in
// [0, 1]. Digits, spaces, punctuation and non-ASCII letters are not counted
// on either side. Text with no English letters scores 0.
export function allcaps(text: string): number {
  const letters = text.match(/[A-Za-z]/g)?.length ?? 0;
  if (letters === 0) return 0;
  const upper = text.match(/[A-Z]/g)?.length ?? 0;
  return upper / letters;
}
