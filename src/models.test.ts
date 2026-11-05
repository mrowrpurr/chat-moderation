import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { path7za } from "7zip-bin";
import { ensureModel, modelsDir, modelsUrl, MODEL_SOURCES, type ModelSource } from "./models";
import { MODELS } from "./test-models";

const work = join(tmpdir(), `chat-moderation-models-test-${process.pid}`);
const name = "tiny-model-q8";
let server: ReturnType<typeof Bun.serve>;
let serve: () => ReturnType<typeof Bun.serve>;
let source: ModelSource;
let baseUrl: string;

beforeAll(async () => {
  // A small stand-in folder: real tokenizer and config files, a fake model.onnx.
  const src = join(work, "src", name);
  mkdirSync(src, { recursive: true });
  const real = join(MODELS, "unbiased-toxic-roberta-detoxify-q8");
  for (const f of ["config.json", "tokenizer.json", "tokenizer_config.json"]) cpSync(join(real, f), join(src, f));
  writeFileSync(join(src, "model.onnx"), "not really onnx");
  const archive = join(work, `${name}.7z`);
  const proc = Bun.spawn([path7za, "a", "-t7z", "-bso0", "-bsp0", archive, name], { cwd: join(work, "src") });
  expect(await proc.exited).toBe(0);
  const bytes = readFileSync(archive);
  source = { archive: `${name}.7z`, bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") };
  const handler = (req: Request) => (new URL(req.url).pathname === `/${name}.7z` ? new Response(bytes) : new Response("nope", { status: 404 }));
  server = Bun.serve({ port: 0, fetch: handler });
  const port = server.port;
  serve = () => Bun.serve({ port, fetch: handler });
  baseUrl = `http://localhost:${port}/`;
});

afterAll(() => {
  server?.stop(true);
  rmSync(work, { recursive: true, force: true });
});

describe("modelsDir", () => {
  test("a per-user folder ending in chat-moderation/models, never inside the package", () => {
    expect(modelsDir().replace(/\\/g, "/")).toEndWith("chat-moderation/models");
    expect(modelsDir()).not.toContain("node_modules");
  });
});

describe("modelsUrl", () => {
  test("the option, else models.json; neither set is an error", () => {
    expect(modelsUrl("https://x/y")).toBe("https://x/y/");
    expect(modelsUrl()).toBe(MODEL_SOURCES.baseUrl);
    const saved = MODEL_SOURCES.baseUrl;
    MODEL_SOURCES.baseUrl = "";
    try {
      expect(() => modelsUrl()).toThrow("No models URL");
    } finally {
      MODEL_SOURCES.baseUrl = saved;
    }
  });
});

describe("ensureModel", () => {
  test("downloads, verifies, extracts, and is a no-op the second time", async () => {
    const dir = join(work, "cache-ok");
    const got = await ensureModel(name, { modelsDir: dir, modelsUrl: baseUrl, sources: { [name]: source } });
    expect(got).toBe(join(dir, name));
    for (const f of ["model.onnx", "config.json", "tokenizer.json", "tokenizer_config.json"]) expect(existsSync(join(got, f))).toBe(true);
    expect(existsSync(join(dir, ".tmp", source.archive))).toBe(false);
    server.stop(true);
    try {
      expect(await ensureModel(name, { modelsDir: dir, modelsUrl: baseUrl, sources: { [name]: source } })).toBe(got);
    } finally {
      server = serve();
    }
  }, 30_000);

  test("a wrong hash leaves nothing behind and throws", async () => {
    const dir = join(work, "cache-bad");
    const bad = { ...source, sha256: "0".repeat(64) };
    await expect(ensureModel(name, { modelsDir: dir, modelsUrl: baseUrl, sources: { [name]: bad } })).rejects.toThrow("failed verification");
    expect(existsSync(join(dir, name))).toBe(false);
    expect(existsSync(join(dir, ".tmp", source.archive))).toBe(false);
  }, 30_000);

  test("unknown model with no source is an error naming the folder", async () => {
    await expect(ensureModel("nope-fp32", { modelsDir: join(work, "cache-none") })).rejects.toThrow("nope-fp32");
  });
});
