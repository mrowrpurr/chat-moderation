import { afterAll, beforeAll, describe, expect, test } from "./testing.ts";
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer, type RequestListener, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { path7za } from "7zip-bin";
import { ensureModel, modelsDir, modelsUrl, MODEL_SOURCES, type ModelSource } from "./models.ts";
import { MODELS } from "./test-models.ts";

const work = join(tmpdir(), `chat-moderation-models-test-${process.pid}`);
const name = "tiny-model-q8";
let server: Server;
let serve: () => Promise<Server>;
let source: ModelSource;
let baseUrl: string;

const listen = (handler: RequestListener, port: number) =>
  new Promise<Server>((resolve) => {
    const s = createServer(handler);
    s.listen(port, "127.0.0.1", () => resolve(s));
  });
const stop = (s: Server) =>
  new Promise<void>((resolve) => {
    s.closeAllConnections();
    s.close(() => resolve());
  });

beforeAll(async () => {
  // A small stand-in folder: real tokenizer and config files, a fake model.onnx.
  const src = join(work, "src", name);
  mkdirSync(src, { recursive: true });
  const real = join(MODELS, "unbiased-toxic-roberta-detoxify-q8");
  for (const f of ["config.json", "tokenizer.json", "tokenizer_config.json"]) cpSync(join(real, f), join(src, f));
  writeFileSync(join(src, "model.onnx"), "not really onnx");
  const archive = join(work, `${name}.7z`);
  await promisify(execFile)(path7za, ["a", "-t7z", "-bso0", "-bsp0", archive, name], { cwd: join(work, "src") });
  const bytes = readFileSync(archive);
  source = { archive: `${name}.7z`, bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") };
  const handler: RequestListener = (req, res) => {
    if (req.url === `/${name}.7z`) {
      res.writeHead(200, { "content-type": "application/octet-stream", "content-length": bytes.length });
      res.end(bytes);
    } else {
      res.writeHead(404);
      res.end("nope");
    }
  };
  server = await listen(handler, 0);
  const port = (server.address() as AddressInfo).port;
  serve = () => listen(handler, port);
  baseUrl = `http://127.0.0.1:${port}/`;
});

afterAll(async () => {
  if (server) await stop(server);
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
    await stop(server);
    try {
      expect(await ensureModel(name, { modelsDir: dir, modelsUrl: baseUrl, sources: { [name]: source } })).toBe(got);
    } finally {
      server = await serve();
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
