// Test runner shim: the same describe/test/hooks/expect under `bun test` and `node --test`.
// Under Bun it is bun:test. Under Node it is node:test plus the handful of matchers the suite uses.
type TestFn = () => void | Promise<void>;
type HookFn = () => void | Promise<void>;

export type Matchers = {
  toBe(expected: unknown): void;
  toEqual(expected: unknown): void;
  toBeCloseTo(expected: number, digits?: number): void;
  toBeGreaterThan(expected: number): void;
  toBeGreaterThanOrEqual(expected: number): void;
  toBeLessThan(expected: number): void;
  toBeLessThanOrEqual(expected: number): void;
  toContain(expected: string): void;
  toEndWith(expected: string): void;
  toHaveLength(expected: number): void;
  toThrow(message?: string): void;
  not: Matchers;
  rejects: { toThrow(message?: string): Promise<void> };
};

export type Testing = {
  describe(name: string, fn: () => void): void;
  test(name: string, fn: TestFn, timeoutMs?: number): void;
  beforeAll(fn: HookFn): void;
  afterAll(fn: HookFn): void;
  expect(received: unknown): Matchers;
};

async function bunTesting(): Promise<Testing> {
  const b = await import("bun:test");
  return { describe: b.describe, test: b.test, beforeAll: b.beforeAll, afterAll: b.afterAll, expect: b.expect as unknown as Testing["expect"] };
}

async function nodeTesting(): Promise<Testing> {
  const n = await import("node:test");
  const assert = (await import("node:assert/strict")).default;
  const fail = (msg: string) => new assert.AssertionError({ message: msg });
  const closeTo = (a: number, b: number, digits: number) => Math.abs(a - b) < 10 ** -digits / 2;
  const messageOf = (e: unknown) => (e instanceof Error ? e.message : String(e));
  const matchers = (received: unknown, negated = false): Matchers => {
    const check = (ok: boolean, msg: string) => {
      if (ok === negated) throw fail(`${negated ? "not " : ""}${msg}`);
    };
    const num = received as number;
    const str = received as string;
    const list = received as { length: number; includes(x: unknown): boolean };
    const fn = received as () => unknown;
    const m: Matchers = {
      toBe: (e) => (negated ? assert.notStrictEqual(received, e) : assert.strictEqual(received, e)),
      toEqual: (e) => (negated ? assert.notDeepStrictEqual(received, e) : assert.deepStrictEqual(received, e)),
      toBeCloseTo: (e, digits = 2) => check(closeTo(num, e, digits), `expected ${received} to be close to ${e} (${digits} digits)`),
      toBeGreaterThan: (e) => check(num > e, `expected ${received} > ${e}`),
      toBeGreaterThanOrEqual: (e) => check(num >= e, `expected ${received} >= ${e}`),
      toBeLessThan: (e) => check(num < e, `expected ${received} < ${e}`),
      toBeLessThanOrEqual: (e) => check(num <= e, `expected ${received} <= ${e}`),
      toContain: (e) => check(list.includes(e), `expected ${JSON.stringify(received)} to contain ${JSON.stringify(e)}`),
      toEndWith: (e) => check(str.endsWith(e), `expected ${JSON.stringify(received)} to end with ${JSON.stringify(e)}`),
      toHaveLength: (e) => check(list.length === e, `expected length ${e}, got ${list.length}`),
      toThrow: (message) => {
        let threw = false;
        let got = "";
        try {
          fn();
        } catch (e) {
          threw = true;
          got = messageOf(e);
        }
        check(threw && (message === undefined || got.includes(message)), `expected to throw ${message ?? ""}, got ${threw ? JSON.stringify(got) : "no throw"}`);
      },
      get not() {
        return matchers(received, !negated);
      },
      rejects: {
        toThrow: async (message) => {
          let threw = false;
          let got = "";
          try {
            await (received as Promise<unknown>);
          } catch (e) {
            threw = true;
            got = messageOf(e);
          }
          check(threw && (message === undefined || got.includes(message)), `expected rejection ${message ?? ""}, got ${threw ? JSON.stringify(got) : "no rejection"}`);
        },
      },
    };
    return m;
  };
  return {
    describe: (name, fn) => void n.describe(name, fn),
    test: (name, fn, timeoutMs) => void n.test(name, timeoutMs ? { timeout: timeoutMs } : {}, fn),
    beforeAll: (fn) => n.before(fn),
    afterAll: (fn) => n.after(fn),
    expect: (received) => matchers(received),
  };
}

export const { describe, test, beforeAll, afterAll, expect }: Testing = await (process.versions.bun === undefined ? nodeTesting() : bunTesting());
