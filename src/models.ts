// Where models live on disk, and fetching one that is not there yet. 🍞
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream, existsSync } from "node:fs";
import { chmod, mkdir, rename, rm, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { setTimeout as sleep } from "node:timers/promises";
import { promisify } from "node:util";
import sources from "./models.json" with { type: "json" };

export type ModelSource = { archive: string; bytes: number; sha256: string };

/** Archive name, size and sha256 for every model folder the library knows; `baseUrl` is where they are published. */
export const MODEL_SOURCES: { baseUrl: string; models: Record<string, ModelSource> } = sources;

/**
 * The per-user folder model folders are read from and downloaded into, unless
 * the `modelsDir` option says otherwise.
 *
 *   macOS    ~/Library/Caches/chat-moderation/models
 *   Linux    $XDG_CACHE_HOME/chat-moderation/models, else ~/.cache/chat-moderation/models
 *   Windows  %LOCALAPPDATA%\chat-moderation\models
 */
export function modelsDir(): string {
  switch (process.platform) {
    case "darwin":
      return join(homedir(), "Library", "Caches", "chat-moderation", "models");
    case "win32":
      return join(process.env.LOCALAPPDATA ?? join(homedir(), "AppData", "Local"), "chat-moderation", "models");
    default:
      return join(process.env.XDG_CACHE_HOME ?? join(homedir(), ".cache"), "chat-moderation", "models");
  }
}

/** Where archives are downloaded from: the `modelsUrl` option, else `baseUrl` in `models.json`. */
export function modelsUrl(option?: string): string {
  const url = option ?? MODEL_SOURCES.baseUrl;
  if (!url) {
    throw new Error(`No models URL: pass modelsUrl, or put the model folders in ${modelsDir()} or the modelsDir you pass.`);
  }
  return url.endsWith("/") ? url : `${url}/`;
}

export type EnsureOptions = {
  modelsDir?: string;
  modelsUrl?: string;
  sources?: Record<string, ModelSource>;
};

/**
 * Returns the folder for `name` (`<model>-<precision>`), downloading, verifying
 * and extracting its archive first if `model.onnx` is not already there.
 */
export async function ensureModel(name: string, options: EnsureOptions = {}): Promise<string> {
  const base = options.modelsDir ? resolve(options.modelsDir) : modelsDir();
  const dir = join(base, name);
  if (existsSync(join(dir, "model.onnx"))) return dir;

  const source = (options.sources ?? MODEL_SOURCES.models)[name];
  if (!source) throw new Error(`No model folder at ${dir} and no download source for ${name}.`);
  const url = `${modelsUrl(options.modelsUrl)}${source.archive}`;

  const tmp = join(base, ".tmp");
  await mkdir(tmp, { recursive: true });
  await withLock(join(tmp, `${name}.lock`), async () => {
    if (existsSync(join(dir, "model.onnx"))) return; // another process finished it while we waited
    const archive = join(tmp, source.archive);
    const staging = join(tmp, name);
    await rm(staging, { recursive: true, force: true });
    await download(url, archive);
    await verify(archive, source);
    await extract(archive, tmp);
    await rm(archive, { force: true });
    await rm(dir, { recursive: true, force: true });
    await rename(staging, dir);
  });
  return dir;
}

async function download(url: string, to: string): Promise<void> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Download failed: ${res.status} ${res.statusText} for ${url}`);
  await pipeline(Readable.fromWeb(res.body as import("node:stream/web").ReadableStream), createWriteStream(to));
}

async function verify(archive: string, source: ModelSource): Promise<void> {
  const size = (await stat(archive)).size;
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(archive)) hash.update(chunk);
  const sha256 = hash.digest("hex");
  if (size !== source.bytes || sha256 !== source.sha256) {
    await rm(archive, { force: true });
    throw new Error(
      `${source.archive} failed verification: got ${size} bytes sha256 ${sha256}, expected ${source.bytes} bytes sha256 ${source.sha256}.`,
    );
  }
}

async function extract(archive: string, into: string): Promise<void> {
  const { path7za } = await import("7zip-bin");
  await chmod(path7za, 0o755).catch(() => {}); // package installs drop the executable bit
  try {
    await promisify(execFile)(path7za, ["x", "-y", `-o${into}`, "-bso0", "-bsp0", archive]);
  } catch (error) {
    const stderr = (error as { stderr?: string }).stderr ?? "";
    throw new Error(`7z failed on ${archive}: ${stderr || String(error)}`);
  }
}

/** Exclusive-mkdir lock so two processes do not fetch the same model at once. */
async function withLock(lockDir: string, fn: () => Promise<void>): Promise<void> {
  const started = Date.now();
  for (;;) {
    try {
      await mkdir(lockDir);
      break;
    } catch {
      if (Date.now() - started > 30 * 60_000) throw new Error(`Timed out waiting for ${lockDir}; remove it if no download is running.`);
      await sleep(500);
    }
  }
  try {
    await fn();
  } finally {
    await rm(lockDir, { recursive: true, force: true });
  }
}
