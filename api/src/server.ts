import Fastify, { type FastifyInstance } from "fastify";
import multipart from "@fastify/multipart";
import { DatabaseSync } from "node:sqlite";
import { execFile as execFileCallback } from "node:child_process";
import { createReadStream, createWriteStream } from "node:fs";
import { access, mkdir, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { cpus, freemem, loadavg, totalmem } from "node:os";
import { basename, dirname, extname, isAbsolute, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline } from "node:stream/promises";
import { promisify } from "node:util";
import { randomUUID } from "node:crypto";

const execFile = promisify(execFileCallback);
const apiDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const projectRoot = resolve(apiDir, "..");
const databasePath = resolve(process.env.TRANSLATOR_DB ?? resolve(projectRoot, "data/translator.sqlite3"));
const pythonPath = resolve(process.env.TRANSLATOR_PYTHON ?? resolve(projectRoot, ".venv/bin/python"));
const queueScript = resolve(projectRoot, "job_queue.py");
const reviewScript = resolve(projectRoot, "review_render.py");
const jobsDirectory = resolve(process.env.TRANSLATOR_JOBS_DIR ?? resolve(projectRoot, "data/jobs"));
const uploadDirectory = resolve(projectRoot, "data/incoming");
const memoryAnalysisDirectory = resolve(projectRoot, "data/memory-analysis");
const allowedMime = new Map([
  ["image/png", ".png"], ["image/jpeg", ".jpg"], ["image/webp", ".webp"],
  ["application/zip", ".zip"], ["application/x-zip", ".zip"], ["application/x-zip-compressed", ".zip"],
]);

type SqlRow = Record<string, unknown>;

function parseJob(row: SqlRow | undefined): SqlRow | undefined {
  if (!row) return undefined;
  const result = { ...row };
  result.options = JSON.parse(String(result.options_json ?? "{}"));
  delete result.options_json;
  return result;
}

export type QueueMove = "up" | "down" | "top";

/**
 * Apply one manual reorder to the waiting queue.
 *
 * Only the relative order of the pending jobs matters: the worker always takes
 * the smallest `queue_position`, so the returned list is renumbered 0..n-1 in
 * the requested order. A pure function so the button behaviour is unit-tested
 * without a live database. Returns null when the job is not in the queue.
 */
export function reorderQueue(
  entries: Array<{ id: string; position: number }>,
  jobId: string,
  move: QueueMove,
): Array<{ id: string; position: number }> | null {
  const index = entries.findIndex((entry) => entry.id === jobId);
  if (index < 0) return null;
  const next = entries.slice();
  const [row] = next.splice(index, 1);
  if (!row) return null;
  const target = move === "top" ? 0 : move === "up" ? Math.max(0, index - 1) : Math.min(next.length, index + 1);
  next.splice(target, 0, row);
  return next.map((entry, order) => ({ id: entry.id, position: order }));
}

function openDatabase(): DatabaseSync {
  const database = new DatabaseSync(databasePath);
  database.exec("PRAGMA foreign_keys=ON; PRAGMA busy_timeout=30000;");
  return database;
}

function folderSlug(value: string, fallback: string): string {
  return value.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80) || fallback;
}

async function pathExists(path: string): Promise<boolean> {
  try { await access(path); return true; } catch { return false; }
}

export function jobDownloadFilename(row: SqlRow): string {
  let options: Record<string, unknown> = {};
  try { options = JSON.parse(String(row.options_json ?? "{}")); } catch { /* legacy job */ }
  const output = String(row.output_path ?? "translated.zip");
  const suffix = extname(output).toLowerCase() || ".zip";
  const title = folderSlug(String(options.title_slug || options.comic_title || "komik"), "komik");
  const rawChapter = String(options.chapter || "1").replace(/^\s*(?:chapter|ch)\s*[-_. ]*/i, "");
  return `${title}-ch-${folderSlug(rawChapter, "1")}${suffix}`;
}

/**
 * Reduce a stage subprocess failure to the one line that explains it.
 *
 * `execFile` rejects with `Command failed: <cmd>\n<stderr>`, so surfacing
 * `error.message` verbatim puts the whole command line and a Python traceback
 * into the dashboard toast. The pipeline raises its real reason as the last
 * `ValueError: ...` line, which is what the operator needs to act on.
 */
export function stageErrorMessage(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error ?? "");
  const lines = raw.split("\n").map((line) => line.trim()).filter(Boolean);
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const match = /^(?:\w*Error|Exception):\s*(.+)$/.exec(lines[index] ?? "");
    const reason = match?.[1]?.trim();
    if (reason) return reason;
  }
  const last = lines.at(-1) ?? "";
  if (last.startsWith("Command failed:")) return "Render gagal dijalankan.";
  return last || "Render gagal.";
}

/**
 * Rebuild a job's chapter WebP archive from its rendered PNGs.
 *
 * `review_render.py archive --db` reads the per-page crops from the database, so
 * the ZIP always reflects the trims the operator set in the editor. A failure is
 * logged and reported, never thrown: a stale archive must not block the crop
 * itself from being saved.
 */
async function rebuildArchive(jobOutput: string, log: { warn: (value: unknown) => void }): Promise<boolean> {
  if (extname(jobOutput).toLowerCase() !== ".zip") return false;
  try {
    await execFile(pythonPath, [reviewScript, "archive", dirname(jobOutput), jobOutput, "--db", databasePath], { cwd: projectRoot, maxBuffer: 1024 * 1024 });
    return true;
  } catch (error) {
    log.warn(error);
    return false;
  }
}

/**
 * Delete every cached preview for one page.
 *
 * Previews are cached per requested width (`{imageId}@{width}.jpg`), so anything
 * that changes the rendered pixels — a re-render, or a new crop — invalidates all
 * width variants, not just one.
 */
async function clearPreviews(outputPath: string, imageId: string): Promise<void> {
  const previewDir = resolve(dirname(outputPath), "previews");
  try {
    for (const name of await readdir(previewDir)) {
      if (name.startsWith(`${imageId}@`) || name === `${imageId}.jpg`) {
        await rm(resolve(previewDir, name), { force: true });
      }
    }
  } catch {
    // No previews directory yet: nothing to invalidate.
  }
}

/**
 * Delete the cached single-page download(s) for one page.
 *
 * `GET /api/images/:id/download` caches `<stem>.webp` / `<stem>.zip` beside the
 * rendered page; a crop changes what those files must contain.
 */
async function clearDownloads(outputPath: string): Promise<void> {
  const directory = resolve(dirname(outputPath), "downloads");
  const stem = basename(outputPath, extname(outputPath));
  try {
    for (const name of await readdir(directory)) {
      if (name === `${stem}.webp` || name === `${stem}.zip` || name.startsWith(`${stem}_part`)) {
        await rm(resolve(directory, name), { force: true });
      }
    }
  } catch {
    // No downloads directory yet: nothing to invalidate.
  }
}

/** Crop values for a page, defaulting to no trim. */
function cropOf(row: SqlRow): { cropTop: number; cropBottom: number; cropUpdatedMs: number } {
  return {
    cropTop: Math.max(0, Number(row.crop_top ?? 0) || 0),
    cropBottom: Math.max(0, Number(row.crop_bottom ?? 0) || 0),
    cropUpdatedMs: Math.max(0, Number(row.crop_updated_at ?? 0) || 0) * 1000,
  };
}

/**
 * Validate a requested crop, returning the message to show or `null` when fine.
 *
 * A crop that consumes the whole page would deliver an empty image, so it is
 * rejected with the numbers spelled out — the operator can see exactly which
 * value overshot rather than watching the page vanish.
 */
export function cropError(height: number, cropTop: number, cropBottom: number): string | null {
  if (!Number.isFinite(cropTop) || !Number.isFinite(cropBottom) || cropTop < 0 || cropBottom < 0) {
    return "Nilai potongan harus angka ≥ 0.";
  }
  if (height > 0 && cropTop + cropBottom > height - 1) {
    return `Potongan (${cropTop}+${cropBottom}) melebihi tinggi halaman (${height}px).`;
  }
  return null;
}

function requireToken(request: { headers: Record<string, unknown> }): boolean {
  const expected = process.env.TRANSLATOR_API_TOKEN;
  if (!expected) return true;
  return request.headers.authorization === `Bearer ${expected}`;
}

export async function buildApp(): Promise<FastifyInstance> {
  await mkdir(uploadDirectory, { recursive: true });
  await mkdir(memoryAnalysisDirectory, { recursive: true });
  // Initializes/migrates the schema through the single Python source of truth.
  await execFile(pythonPath, [queueScript, "--db", databasePath, "--data-dir", jobsDirectory, "init"], { cwd: projectRoot });
  const database = openDatabase();
  const app = Fastify({ logger: true, bodyLimit: 2 * 1024 * 1024 });
  await app.register(multipart, {
    limits: { files: 60, fileSize: 100 * 1024 * 1024, fields: 10, parts: 70 },
  });

  app.addHook("onRequest", async (request, reply) => {
    if (!requireToken(request as never)) return reply.code(401).send({ error: "Unauthorized" });
  });
  app.addHook("onClose", async () => database.close());

  app.get("/api/comics", async () => database.prepare(
    "SELECT c.*,count(DISTINCT ch.id) chapter_count,count(DISTINCT j.id) job_count " +
    "FROM comics c LEFT JOIN chapters ch ON ch.comic_id=c.id " +
    "LEFT JOIN jobs j ON json_extract(j.options_json,'$.comic_id')=c.id " +
    "GROUP BY c.id ORDER BY c.updated_at DESC"
  ).all());
  app.post("/api/comics", async (request, reply) => {
    const title = String((request.body as { title?: string })?.title || "").trim().replace(/\s+/g, " ").slice(0, 120);
    if (!title) return reply.code(400).send({ error: "Comic title is required" });
    const slug = folderSlug(title, "comic");
    const existing = database.prepare("SELECT * FROM comics WHERE slug=?").get(slug) as SqlRow | undefined;
    if (existing) return reply.code(409).send({ error: "Comic folder already exists", comic: existing });
    const id = randomUUID(), now = Date.now() / 1000;
    database.prepare("INSERT INTO comics(id,title,slug,created_at,updated_at) VALUES(?,?,?,?,?)").run(id, title, slug, now, now);
    await mkdir(resolve(jobsDirectory, "library", slug), { recursive: true });
    await mkdir(resolve(projectRoot, "data", "memory"), { recursive: true });
    return reply.code(201).send(database.prepare("SELECT * FROM comics WHERE id=?").get(id));
  });
  app.patch("/api/comics/:comicId", async (request, reply) => {
    const { comicId } = request.params as { comicId: string };
    const title = String((request.body as { title?: string })?.title || "").trim().replace(/\s+/g, " ").slice(0, 120);
    if (!title) return reply.code(400).send({ error: "Comic title is required" });
    const comic = database.prepare("SELECT * FROM comics WHERE id=?").get(comicId) as SqlRow | undefined;
    if (!comic) return reply.code(404).send({ error: "Comic folder not found" });
    const slug = folderSlug(title, "comic"), oldSlug = String(comic.slug);
    if (database.prepare("SELECT id FROM comics WHERE slug=? AND id<>?").get(slug, comicId)) return reply.code(409).send({ error: "Comic folder already exists" });
    if (database.prepare("SELECT id FROM jobs WHERE json_extract(options_json,'$.comic_id')=? AND status IN ('QUEUED','WAITING_RESOURCE','PROCESSING') LIMIT 1").get(comicId)) {
      return reply.code(409).send({ error: "Tunggu proses chapter selesai sebelum mengganti nama komik." });
    }
    const libraryRoot = resolve(jobsDirectory, "library"), oldLibrary = resolve(libraryRoot, oldSlug), newLibrary = resolve(libraryRoot, slug);
    const memoryRoot = resolve(projectRoot, "data", "memory");
    const moves: Array<[string, string]> = oldSlug === slug ? [] : [[oldLibrary, newLibrary], [resolve(memoryRoot, oldSlug), resolve(memoryRoot, slug)], [resolve(memoryRoot, `${oldSlug}.json`), resolve(memoryRoot, `${slug}.json`)]];
    for (const [from, to] of moves) if (await pathExists(from) && await pathExists(to)) return reply.code(409).send({ error: `Folder tujuan sudah ada: ${basename(to)}` });
    const completedMoves: Array<[string, string]> = [];
    try {
      await mkdir(libraryRoot, { recursive: true });
      for (const [from, to] of moves) if (await pathExists(from)) { await rename(from, to); completedMoves.push([from, to]); }
      if (!(await pathExists(newLibrary))) await mkdir(newLibrary, { recursive: true });
      database.exec("BEGIN IMMEDIATE");
      database.prepare("UPDATE comics SET title=?,slug=?,updated_at=? WHERE id=?").run(title, slug, Date.now() / 1000, comicId);
      database.prepare("UPDATE jobs SET options_json=json_set(options_json,'$.comic_title',?,'$.title_slug',?),input_path=replace(input_path,?,?),output_path=replace(output_path,?,?),requested_output_path=replace(requested_output_path,?,?) WHERE json_extract(options_json,'$.comic_id')=?").run(title, slug, oldLibrary, newLibrary, oldLibrary, newLibrary, oldLibrary, newLibrary, comicId);
      database.prepare("UPDATE images SET output_path=replace(output_path,?,?),metadata_path=replace(metadata_path,?,?) WHERE job_id IN (SELECT id FROM jobs WHERE json_extract(options_json,'$.comic_id')=?)").run(oldLibrary, newLibrary, oldLibrary, newLibrary, comicId);
      database.exec("COMMIT");
      return database.prepare("SELECT c.*,count(DISTINCT ch.id) chapter_count,count(DISTINCT j.id) job_count FROM comics c LEFT JOIN chapters ch ON ch.comic_id=c.id LEFT JOIN jobs j ON json_extract(j.options_json,'$.comic_id')=c.id WHERE c.id=? GROUP BY c.id").get(comicId);
    } catch (error) {
      try { database.exec("ROLLBACK"); } catch { /* transaction may not have started */ }
      for (const [from, to] of completedMoves.reverse()) if (await pathExists(to)) await rename(to, from);
      request.log.error(error);
      return reply.code(500).send({ error: "Gagal mengganti nama folder komik." });
    }
  });
  app.get("/api/comics/:comicId/chapters", async (request) => {
    const { comicId } = request.params as { comicId: string };
    return database.prepare(
      "SELECT ch.*,count(j.id) job_count FROM chapters ch LEFT JOIN jobs j " +
      "ON json_extract(j.options_json,'$.chapter_id')=ch.id WHERE ch.comic_id=? " +
      "GROUP BY ch.id ORDER BY CAST(ch.number AS REAL),ch.created_at"
    ).all(comicId);
  });
  app.post("/api/comics/:comicId/chapters", async (request, reply) => {
    const { comicId } = request.params as { comicId: string };
    const comic = database.prepare("SELECT * FROM comics WHERE id=?").get(comicId) as SqlRow | undefined;
    if (!comic) return reply.code(404).send({ error: "Comic folder not found" });
    const number = String((request.body as { number?: string })?.number || "").trim().replace(/^chapter\s*/i, "").slice(0, 40);
    if (!number) return reply.code(400).send({ error: "Chapter number is required" });
    const slug = folderSlug(number, "1");
    const existing = database.prepare("SELECT * FROM chapters WHERE comic_id=? AND slug=?").get(comicId, slug) as SqlRow | undefined;
    if (existing) return reply.code(409).send({ error: "Chapter folder already exists", chapter: existing });
    const id = randomUUID(), now = Date.now() / 1000;
    database.prepare("INSERT INTO chapters(id,comic_id,number,slug,created_at,updated_at) VALUES(?,?,?,?,?,?)").run(id, comicId, number, slug, now, now);
    database.prepare("UPDATE comics SET updated_at=? WHERE id=?").run(now, comicId);
    await mkdir(resolve(jobsDirectory, "library", String(comic.slug), `chapter-${slug}`), { recursive: true });
    return reply.code(201).send(database.prepare("SELECT * FROM chapters WHERE id=?").get(id));
  });

  app.get("/api/comics/:comicId/memory", async (request, reply) => {
    const { comicId } = request.params as { comicId: string };
    const comic = database.prepare("SELECT * FROM comics WHERE id=?").get(comicId) as SqlRow | undefined;
    if (!comic) return reply.code(404).send({ error: "Comic folder not found" });
    const folder = resolve(projectRoot, "data", "memory", String(comic.slug));
    let profile: Record<string, unknown> = {};
    try { profile = JSON.parse(await readFile(resolve(folder, "memory-summary.json"), "utf8")); } catch { /* no approved memory */ }
    const references = database.prepare("SELECT * FROM memory_references WHERE comic_id=? ORDER BY created_at DESC").all(comicId).map((row) => {
      const item = { ...(row as SqlRow) }; item.summary = JSON.parse(String(item.summary_json || "{}")); delete item.summary_json; return item;
    });
    const jobs = database.prepare("SELECT id,chapter_number,status,stage,progress,error,proposal_json,created_at,updated_at FROM memory_jobs WHERE comic_id=? ORDER BY created_at DESC LIMIT 10").all(comicId).map((row) => {
      const item = { ...(row as SqlRow) }; item.proposal = item.proposal_json ? JSON.parse(String(item.proposal_json)) : null; delete item.proposal_json; return item;
    });
    return { comic, profile, references, jobs };
  });

  app.post("/api/comics/:comicId/memory/analyze", async (request, reply) => {
    const { comicId } = request.params as { comicId: string };
    const comic = database.prepare("SELECT * FROM comics WHERE id=?").get(comicId) as SqlRow | undefined;
    if (!comic) return reply.code(404).send({ error: "Comic folder not found" });
    const id = randomUUID(), work = resolve(memoryAnalysisDirectory, id);
    await mkdir(work, { recursive: true });
    let rawPath = "", translatedPath = "", chapterNumber = "";
    try {
      for await (const part of request.parts()) {
        if (part.type === "file") {
          if (!["rawZip", "translatedZip"].includes(part.fieldname) || extname(part.filename || "").toLowerCase() !== ".zip") throw new Error("Upload rawZip dan translatedZip dalam format ZIP");
          const destination = resolve(work, `${part.fieldname}.zip`);
          await pipeline(part.file, createWriteStream(destination, { flags: "wx" }));
          if (part.file.truncated) throw new Error(`${part.fieldname} exceeds the upload size limit`);
          if (part.fieldname === "rawZip") rawPath = destination; else translatedPath = destination;
        } else if (part.fieldname === "chapterNumber") chapterNumber = String(part.value).trim().slice(0, 40);
      }
      if (!rawPath || !translatedPath || !chapterNumber) throw new Error("Nomor chapter, ZIP raw, dan ZIP Indonesia wajib diisi");
      const now = Date.now() / 1000;
      database.prepare("INSERT INTO memory_jobs(id,comic_id,chapter_number,status,stage,raw_zip_path,translated_zip_path,work_path,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)")
        .run(id, comicId, chapterNumber, "QUEUED", "QUEUED", rawPath, translatedPath, work, now, now);
      return reply.code(202).send({ id, comic_id: comicId, chapter_number: chapterNumber, status: "QUEUED", stage: "QUEUED", progress: 0 });
    } catch (error) {
      await rm(work, { recursive: true, force: true });
      return reply.code(400).send({ error: error instanceof Error ? error.message : "Invalid memory upload" });
    }
  });

  app.patch("/api/comics/:comicId/memory/jobs/:jobId", async (request, reply) => {
    const { comicId, jobId } = request.params as { comicId: string; jobId: string };
    const row = database.prepare("SELECT * FROM memory_jobs WHERE id=? AND comic_id=?").get(jobId, comicId) as SqlRow | undefined;
    if (!row || String(row.status) !== "WAITING_APPROVAL") return reply.code(409).send({ error: "Memory proposal is not editable" });
    const proposal = (request.body as { proposal?: unknown })?.proposal;
    if (!proposal || typeof proposal !== "object") return reply.code(400).send({ error: "Valid proposal is required" });
    database.prepare("UPDATE memory_jobs SET proposal_json=?,updated_at=? WHERE id=?").run(JSON.stringify(proposal), Date.now() / 1000, jobId);
    return { id: jobId, proposal };
  });

  app.post("/api/comics/:comicId/memory/jobs/:jobId/approve", async (request, reply) => {
    const { comicId, jobId } = request.params as { comicId: string; jobId: string };
    const comic = database.prepare("SELECT * FROM comics WHERE id=?").get(comicId) as SqlRow | undefined;
    const row = database.prepare("SELECT * FROM memory_jobs WHERE id=? AND comic_id=?").get(jobId, comicId) as SqlRow | undefined;
    if (!comic || !row || String(row.status) !== "WAITING_APPROVAL" || !row.proposal_json) return reply.code(409).send({ error: "Memory proposal is not ready" });
    const proposal = JSON.parse(String(row.proposal_json)) as Record<string, unknown>;
    const folder = resolve(projectRoot, "data", "memory", String(comic.slug));
    await mkdir(resolve(folder, "references"), { recursive: true });
    const terminology = Array.isArray(proposal.terminology) ? proposal.terminology as Array<Record<string, unknown>> : [];
    const compatibility: Record<string, string> = {};
    for (const item of terminology) if (item.source && item.target) compatibility[String(item.source)] = String(item.target);
    let previous: Record<string, string> = {};
    try { previous = JSON.parse(await readFile(resolve(projectRoot, "data", "memory", `${comic.slug}.json`), "utf8")); } catch { /* first memory */ }
    await writeFile(resolve(projectRoot, "data", "memory", `${comic.slug}.json`), JSON.stringify({ ...compatibility, ...previous }, null, 2), "utf8");
    await writeFile(resolve(folder, "terminology.json"), JSON.stringify(terminology, null, 2), "utf8");
    await writeFile(resolve(folder, "character-voices.json"), JSON.stringify(proposal.character_voices || [], null, 2), "utf8");
    await writeFile(resolve(folder, "style-profile.json"), JSON.stringify(proposal.style_profile || {}, null, 2), "utf8");
    await writeFile(resolve(folder, "translation-examples.json"), JSON.stringify(proposal.translation_examples || [], null, 2), "utf8");
    await writeFile(resolve(folder, "memory-summary.json"), JSON.stringify(proposal, null, 2), "utf8");
    await writeFile(resolve(folder, "references", `chapter-${folderSlug(String(row.chapter_number), "reference")}.json`), JSON.stringify(proposal, null, 2), "utf8");
    const referenceId = randomUUID(), now = Date.now() / 1000;
    database.exec("BEGIN IMMEDIATE");
    try {
      database.prepare("INSERT INTO memory_references(id,comic_id,memory_job_id,chapter_number,summary_json,created_at) VALUES(?,?,?,?,?,?)").run(referenceId, comicId, jobId, String(row.chapter_number), JSON.stringify({ pair_count: proposal.pair_count, summary: proposal.summary }), now);
      database.prepare("UPDATE memory_jobs SET status='READY',stage='READY',updated_at=? WHERE id=?").run(now, jobId);
      database.prepare("UPDATE comics SET updated_at=? WHERE id=?").run(now, comicId);
      database.exec("COMMIT");
    } catch (error) { database.exec("ROLLBACK"); throw error; }
    return { id: jobId, status: "READY", reference_id: referenceId, terminology_count: terminology.length };
  });

  app.post("/api/comics/:comicId/memory/jobs/:jobId/cancel", async (request, reply) => {
    const { comicId, jobId } = request.params as { comicId: string; jobId: string };
    const result = database.prepare("UPDATE memory_jobs SET status='CANCELLED',stage='CANCELLED',updated_at=?,completed_at=? WHERE id=? AND comic_id=? AND status='QUEUED'")
      .run(Date.now() / 1000, Date.now() / 1000, jobId, comicId);
    if (result.changes !== 1) return reply.code(409).send({ error: "Analysis can no longer be cancelled" });
    return { id: jobId, status: "CANCELLED" };
  });

  app.delete("/api/comics/:comicId/memory/references/:referenceId", async (request, reply) => {
    const { comicId, referenceId } = request.params as { comicId: string; referenceId: string };
    const row = database.prepare("SELECT * FROM memory_references WHERE id=? AND comic_id=?").get(referenceId, comicId) as SqlRow | undefined;
    if (!row) return reply.code(404).send({ error: "Memory reference not found" });
    database.prepare("DELETE FROM memory_references WHERE id=?").run(referenceId);
    return { id: referenceId, status: "DELETED", note: "Approved canonical memory remains until edited or replaced" };
  });

  app.post("/api/jobs", async (request, reply) => {
    const temporaries: string[] = [];
    const originalNames: string[] = [];
    const fields: Record<string, string> = {};
    try {
      for await (const part of request.parts()) {
        if (part.type === "file") {
          const originalName = basename(part.filename || "upload").replace(/[^a-zA-Z0-9._-]/g, "_");
          const declaredSuffix = extname(originalName).toLowerCase();
          const suffix = allowedMime.get(part.mimetype) ?? (
            part.mimetype === "application/octet-stream" && [".png", ".jpg", ".jpeg", ".webp", ".zip"].includes(declaredSuffix)
              ? declaredSuffix === ".jpeg" ? ".jpg" : declaredSuffix
              : undefined
          );
          if (!suffix) throw new Error("Only PNG, JPEG, WebP, and ZIP files are supported");
          if (!originalName.toLowerCase().endsWith(suffix) && !(suffix === ".jpg" && originalName.toLowerCase().endsWith(".jpeg"))) {
            throw new Error("File extension does not match MIME type");
          }
          const temporary = resolve(uploadDirectory, `${randomUUID()}${suffix}`);
          temporaries.push(temporary);
          originalNames.push(originalName);
          await pipeline(part.file, createWriteStream(temporary, { flags: "wx" }));
          if (part.file.truncated) throw new Error(`${originalName} exceeds the upload size limit`);
        } else {
          fields[part.fieldname] = String(part.value);
        }
      }
      if (!temporaries.length) return reply.code(400).send({ error: "At least one image or ZIP is required" });
      const comic = fields.comicId ? database.prepare("SELECT * FROM comics WHERE id=?").get(fields.comicId) as SqlRow | undefined : undefined;
      const chapterRow = fields.chapterId ? database.prepare("SELECT * FROM chapters WHERE id=? AND comic_id=?").get(fields.chapterId, String(fields.comicId || "")) as SqlRow | undefined : undefined;
      if ((fields.comicId || fields.chapterId) && (!comic || !chapterRow)) return reply.code(400).send({ error: "Select a valid comic and chapter folder" });
      const selectedTitle = comic ? String(comic.title) : (fields.title || "Tanpa Judul");
      const selectedChapter = chapterRow ? String(chapterRow.number) : (fields.chapter || "1");
      const command = [
        queueScript, "--db", databasePath, "--data-dir", jobsDirectory, "enqueue", ...temporaries,
        "--provider", fields.provider || "9router", "--env-file", fields.envFile || ".env.9router",
        "--min-y", String(Number.parseInt(fields.minY || "0", 10) || 0),
        "--title", selectedTitle.trim().slice(0, 120),
        "--chapter", selectedChapter.trim().slice(0, 40),
      ];
      const { stdout } = await execFile(pythonPath, command, { cwd: projectRoot, maxBuffer: 1024 * 1024 });
      const job = JSON.parse(stdout);
      // Keep a safe display filename; storage itself is UUID-isolated.
      job.options.original_filename = originalNames.length === 1 ? originalNames[0] : `${originalNames.length} images`;
      job.options.original_names = originalNames;
      job.options.pipeline_engine = fields.pipelineEngine || "v5";
      job.options.hybrid_mode = fields.hybridMode || "xianscan-v3";
      job.options.layout_engine = fields.layoutEngine || "v5";
      if (comic && chapterRow) {
        job.options.comic_id = String(comic.id);
        job.options.chapter_id = String(chapterRow.id);
      }
      database.prepare("UPDATE jobs SET options_json=? WHERE id=?").run(JSON.stringify(job.options), job.id);
      return reply.code(201).send(job);
    } catch (error) {
      request.log.warn(error);
      return reply.code(400).send({ error: error instanceof Error ? error.message : "Invalid upload" });
    } finally {
      await Promise.all(temporaries.map((temporary) => rm(temporary, { force: true })));
    }
  });

  app.get("/api/jobs", async (request) => {
    const query = request.query as { limit?: string };
    const limit = Math.min(100, Math.max(1, Number.parseInt(query.limit || "20", 10) || 20));
    const jobs = database.prepare("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?").all(limit)
      .map((row) => parseJob(row as SqlRow) as SqlRow);
    const reviewRows = database.prepare(
      "SELECT ranked.job_id,ranked.page_number,count(r.id) review_count FROM (" +
      "SELECT id,job_id,row_number() OVER (PARTITION BY job_id ORDER BY filename) page_number FROM images" +
      ") ranked JOIN regions r ON r.image_id=ranked.id WHERE r.status='NEEDS_REVIEW' " +
      "GROUP BY ranked.job_id,ranked.page_number ORDER BY ranked.page_number"
    ).all() as SqlRow[];
    for (const job of jobs) job.review_pages = reviewRows
      .filter((row) => String(row.job_id) === String(job.id))
      .map((row) => ({ page: Number(row.page_number), count: Number(row.review_count) }));
    return jobs;
  });
  app.get("/api/jobs/:jobId", async (request, reply) => {
    const { jobId } = request.params as { jobId: string };
    const job = parseJob(database.prepare("SELECT * FROM jobs WHERE id=?").get(jobId) as SqlRow | undefined);
    return job ?? reply.code(404).send({ error: "Job not found" });
  });
  app.post("/api/jobs/:jobId/cancel", async (request, reply) => {
    const { jobId } = request.params as { jobId: string };
    const now = Date.now() / 1000;
    const result = database.prepare(
      "UPDATE jobs SET status='CANCELLED',stage=CASE WHEN status='PROCESSING' THEN 'CANCEL_REQUESTED' ELSE 'CANCELLED' END,updated_at=?,completed_at=? WHERE id=? AND status IN ('QUEUED','WAITING_RESOURCE','PROCESSING')"
    ).run(now, now, jobId);
    if (result.changes !== 1) return reply.code(409).send({ error: "Job is missing or can no longer be stopped" });
    return { id: jobId, status: "CANCELLED" };
  });
  // Manual queue order. The worker picks the smallest queue_position among the
  // waiting jobs, so a reorder simply rewrites those positions. Only QUEUED and
  // WAITING_RESOURCE are movable: a PROCESSING job is already running and a
  // finished one has no place in the queue.
  app.post("/api/jobs/:jobId/queue", async (request, reply) => {
    const { jobId } = request.params as { jobId: string };
    const move = (request.body as { move?: string } | undefined)?.move;
    if (move !== "up" && move !== "down" && move !== "top") {
      return reply.code(400).send({ error: "move harus 'up', 'down', atau 'top'" });
    }
    database.exec("BEGIN IMMEDIATE");
    try {
      const target = database.prepare("SELECT status FROM jobs WHERE id=?").get(jobId) as SqlRow | undefined;
      if (!target) {
        database.exec("ROLLBACK");
        return reply.code(404).send({ error: "Job not found" });
      }
      if (String(target.status) !== "QUEUED" && String(target.status) !== "WAITING_RESOURCE") {
        database.exec("ROLLBACK");
        return reply.code(409).send({ error: "Hanya job yang masih menunggu bisa diatur urutannya" });
      }
      const entries = database.prepare(
        "SELECT id FROM jobs WHERE status IN ('QUEUED','WAITING_RESOURCE') " +
        "ORDER BY COALESCE(queue_position, created_at), created_at"
      ).all() as SqlRow[];
      const ordered = reorderQueue(
        entries.map((row, index) => ({ id: String(row.id), position: index })),
        jobId,
        move,
      );
      if (!ordered) {
        database.exec("ROLLBACK");
        return reply.code(404).send({ error: "Job not found" });
      }
      const update = database.prepare("UPDATE jobs SET queue_position=?, updated_at=? WHERE id=?");
      const now = Date.now() / 1000;
      for (const entry of ordered) update.run(entry.position, now, entry.id);
      database.exec("COMMIT");
      return { id: jobId, move, queue: ordered };
    } catch (error) {
      database.exec("ROLLBACK");
      throw error;
    }
  });
  // Deleting a comic or a chapter is permanent: the row, every child row
  // (chapters, jobs, images, regions, memory) and the whole directory tree are
  // removed, because ON DELETE CASCADE plus a recursive rm covers the lot.
  // Two guards mirror the per-job delete: a running job aborts the request, and
  // the folder must resolve inside jobs/library so a bad slug can never walk
  // out of the storage root.
  const folderSize = async (target: string): Promise<number> => {
    let total = 0;
    const walk = async (current: string) => {
      let entries;
      try {
        entries = await readdir(current, { withFileTypes: true });
      } catch {
        return;
      }
      for (const entry of entries) {
        const child = resolve(current, entry.name);
        if (entry.isDirectory()) await walk(child);
        else {
          try {
            total += (await stat(child)).size;
          } catch {
            /* vanished mid-walk */
          }
        }
      }
    };
    await walk(target);
    return total;
  };

  const previewChapter = (chapterId: string) => {
    const chapter = database.prepare("SELECT * FROM chapters WHERE id=?").get(chapterId) as SqlRow | undefined;
    if (!chapter) return null;
    const comic = database.prepare("SELECT * FROM comics WHERE id=?").get(String(chapter.comic_id)) as SqlRow | undefined;
    const jobs = database.prepare(
      "SELECT id,status FROM jobs WHERE json_extract(options_json,'$.chapter_id')=? OR (json_extract(options_json,'$.comic_id')=? AND json_extract(options_json,'$.chapter')=?)"
    ).all(chapterId, String(chapter.comic_id), String(chapter.number)) as SqlRow[];
    return { chapter, comic, jobs };
  };

  app.get("/api/chapters/:chapterId", async (request, reply) => {
    const found = previewChapter((request.params as { chapterId: string }).chapterId);
    if (!found) return reply.code(404).send({ error: "Chapter not found" });
    const { chapter, comic, jobs } = found;
    const directory = resolve(jobsDirectory, "library", String(comic?.slug ?? ""), `chapter-${String(chapter.slug)}`);
    return {
      id: chapter.id, number: chapter.number, comic_title: comic?.title ?? "",
      job_count: jobs.length, bytes: await folderSize(directory),
      has_running_job: jobs.some((job) => String(job.status) === "PROCESSING"),
    };
  });

  app.delete("/api/chapters/:chapterId", async (request, reply) => {
    const chapterId = (request.params as { chapterId: string }).chapterId;
    const found = previewChapter(chapterId);
    if (!found) return reply.code(404).send({ error: "Chapter not found" });
    const { chapter, comic, jobs } = found;
    if (jobs.some((job) => String(job.status) === "PROCESSING")) {
      return reply.code(409).send({ error: "A chapter with a processing job cannot be deleted" });
    }
    const comicSlug = String(comic?.slug ?? "");
    const directory = resolve(jobsDirectory, "library", comicSlug, `chapter-${String(chapter.slug)}`);
    const relativeDirectory = relative(jobsDirectory, directory);
    if (!relativeDirectory || relativeDirectory.startsWith("..") || isAbsolute(relativeDirectory)) {
      return reply.code(409).send({ error: "Chapter storage path is invalid" });
    }
    const bytes = await folderSize(directory);
    database.exec("BEGIN IMMEDIATE");
    try {
      for (const job of jobs) database.prepare("DELETE FROM jobs WHERE id=?").run(String(job.id));
      database.prepare("DELETE FROM chapters WHERE id=?").run(chapterId);
      database.exec("COMMIT");
    } catch (error) {
      database.exec("ROLLBACK");
      throw error;
    }
    await rm(directory, { recursive: true, force: true });
    return { id: chapterId, status: "DELETED", jobs: jobs.length, bytes };
  });

  const previewComic = (comicId: string) => {
    const comic = database.prepare("SELECT * FROM comics WHERE id=?").get(comicId) as SqlRow | undefined;
    if (!comic) return null;
    const chapters = database.prepare("SELECT id,number,slug FROM chapters WHERE comic_id=?").all(comicId) as SqlRow[];
    const jobs = database.prepare(
      "SELECT id,status FROM jobs WHERE json_extract(options_json,'$.comic_id')=?"
    ).all(comicId) as SqlRow[];
    const memories = database.prepare("SELECT count(*) c FROM memory_jobs WHERE comic_id=?").get(comicId) as SqlRow;
    const references = database.prepare("SELECT count(*) c FROM memory_references WHERE comic_id=?").get(comicId) as SqlRow;
    return { comic, chapters, jobs, memories: Number(memories?.c ?? 0), references: Number(references?.c ?? 0) };
  };

  app.get("/api/comics/:comicId/deletion-preview", async (request, reply) => {
    const found = previewComic((request.params as { comicId: string }).comicId);
    if (!found) return reply.code(404).send({ error: "Comic not found" });
    const directory = resolve(jobsDirectory, "library", String(found.comic.slug));
    return {
      id: found.comic.id, title: found.comic.title,
      chapter_count: found.chapters.length, job_count: found.jobs.length,
      memory_jobs: found.memories, memory_references: found.references,
      bytes: await folderSize(directory),
      has_running_job: found.jobs.some((job) => String(job.status) === "PROCESSING"),
    };
  });

  app.delete("/api/comics/:comicId", async (request, reply) => {
    const comicId = (request.params as { comicId: string }).comicId;
    const found = previewComic(comicId);
    if (!found) return reply.code(404).send({ error: "Comic not found" });
    const { comic, chapters, jobs } = found;
    if (jobs.some((job) => String(job.status) === "PROCESSING")) {
      return reply.code(409).send({ error: "A comic with a processing job cannot be deleted" });
    }
    const directory = resolve(jobsDirectory, "library", String(comic.slug));
    const relativeDirectory = relative(jobsDirectory, directory);
    if (!relativeDirectory || relativeDirectory.startsWith("..") || isAbsolute(relativeDirectory)) {
      return reply.code(409).send({ error: "Comic storage path is invalid" });
    }
    const bytes = await folderSize(directory);
    database.exec("BEGIN IMMEDIATE");
    try {
      for (const job of jobs) database.prepare("DELETE FROM jobs WHERE id=?").run(String(job.id));
      // chapters and every memory row cascade from the comic delete
      database.prepare("DELETE FROM comics WHERE id=?").run(comicId);
      database.exec("COMMIT");
    } catch (error) {
      database.exec("ROLLBACK");
      throw error;
    }
    await rm(directory, { recursive: true, force: true });
    return { id: comicId, status: "DELETED", chapters: chapters.length, jobs: jobs.length, bytes };
  });

  app.delete("/api/jobs/:jobId", async (request, reply) => {
    const { jobId } = request.params as { jobId: string };
    const row = database.prepare("SELECT status,input_path FROM jobs WHERE id=?").get(jobId) as SqlRow | undefined;
    if (!row) return reply.code(404).send({ error: "Job not found" });
    if (String(row.status) === "PROCESSING") return reply.code(409).send({ error: "A processing job cannot be deleted" });
    const jobRoot = resolve(dirname(dirname(String(row.input_path))));
    const relativeJobRoot = relative(jobsDirectory, jobRoot);
    if (!relativeJobRoot || relativeJobRoot.startsWith("..") || isAbsolute(relativeJobRoot)) {
      return reply.code(409).send({ error: "Job storage path is invalid" });
    }
    database.exec("BEGIN IMMEDIATE");
    try {
      const current = database.prepare("SELECT status FROM jobs WHERE id=?").get(jobId) as SqlRow | undefined;
      if (!current) {
        database.exec("ROLLBACK");
        return reply.code(404).send({ error: "Job not found" });
      }
      if (String(current.status) === "PROCESSING") {
        database.exec("ROLLBACK");
        return reply.code(409).send({ error: "A processing job cannot be deleted" });
      }
      database.prepare("DELETE FROM jobs WHERE id=?").run(jobId);
      database.exec("COMMIT");
    } catch (error) {
      database.exec("ROLLBACK");
      throw error;
    }
    await rm(jobRoot, { recursive: true, force: true });
    return { id: jobId, status: "DELETED" };
  });
  app.get("/api/jobs/:jobId/images", async (request) => {
    const { jobId } = request.params as { jobId: string };
    return database.prepare("SELECT * FROM images WHERE job_id=? ORDER BY filename").all(jobId);
  });
  app.get("/api/images/:imageId/preview", async (request, reply) => {
    const { imageId } = request.params as { imageId: string };
    const row = database.prepare("SELECT output_path,crop_top,crop_bottom,crop_updated_at FROM images WHERE id=?").get(imageId) as SqlRow | undefined;
    if (!row) return reply.code(404).send({ error: "Image not found" });
    const source = String(row.output_path);
    // Webtoon review shows every page stacked at full width, so the reader can
    // ask for a wider render than the 420 px single-page default. The width is
    // part of the cache filename; otherwise a wide request would poison the
    // cached thumbnail every other view relies on.
    const requested = Number.parseInt(String((request.query as Record<string, string>).w ?? ""), 10);
    const maxWidth = Number.isFinite(requested) ? Math.min(1600, Math.max(120, requested)) : 420;
    // Page mode edits regions against the FULL page, so its preview must stay
    // uncropped or every region box would be offset by the trim. Webtoon mode is
    // the reading view and asks for `crop=1` to show the delivered result.
    const wantsCrop = String((request.query as Record<string, string>).crop ?? "") === "1";
    const stored = cropOf(row);
    const cropTop = wantsCrop ? stored.cropTop : 0;
    const cropBottom = wantsCrop ? stored.cropBottom : 0;
    const cropSuffix = cropTop || cropBottom ? `-c${cropTop}-${cropBottom}` : "";
    const destination = resolve(dirname(source), "previews", `${imageId}@${maxWidth}${cropSuffix}.jpg`);
    let refreshPreview = false;
    try {
      await access(destination);
      // Regenerate when the rendered page — or the crop — is newer than the
      // cached preview. The crop timestamp is a DB column, not a file mtime, so
      // it is compared explicitly.
      const sourceStat = await stat(source);
      const previewStat = await stat(destination);
      refreshPreview = sourceStat.mtimeMs > previewStat.mtimeMs || (wantsCrop && stored.cropUpdatedMs > previewStat.mtimeMs);
    } catch {
      refreshPreview = true;
    }
    if (refreshPreview) {
      await mkdir(dirname(destination), { recursive: true });
      await execFile(pythonPath, [reviewScript, "preview", source, destination, "--max-width", String(maxWidth), "--crop-top", String(cropTop), "--crop-bottom", String(cropBottom)], { cwd: projectRoot });
    }
    return reply.type("image/jpeg").header("Cache-Control", "no-store").send(createReadStream(destination));
  });
  app.get("/api/images/:imageId/regions", async (request) => {
    const { imageId } = request.params as { imageId: string };
    return database.prepare("SELECT * FROM regions WHERE image_id=? ORDER BY rowid").all(imageId).map((row) => {
      const item = { ...(row as SqlRow) };
      item.bbox = JSON.parse(String(item.bbox_json));
      item.metadata = JSON.parse(String(item.metadata_json));
      delete item.bbox_json;
      delete item.metadata_json;
      return item;
    });
  });
  app.patch("/api/regions/:regionId", async (request, reply) => {
    const { regionId } = request.params as { regionId: string };
    const body = request.body as { translation?: string; x?: number; y?: number; width?: number; height?: number; fontSize?: number; alignment?: string; lineSpacing?: number };
    const translation = body?.translation?.trim();
    if (!translation || translation.length > 2000) return reply.code(400).send({ error: "Valid translation is required" });
    const row = database.prepare("SELECT bbox_json,metadata_json FROM regions WHERE id=?").get(regionId) as SqlRow | undefined;
    if (!row) return reply.code(404).send({ error: "Region not found" });
    const bbox = JSON.parse(String(row.bbox_json)) as { x: number; y: number; width: number; height: number };
    const metadata = JSON.parse(String(row.metadata_json || "{}")) as Record<string, unknown>;
    for (const key of ["x", "y", "width", "height"] as const) {
      if (body[key] !== undefined) bbox[key] = Math.round(Number(body[key]));
    }
    if (bbox.x < 0 || bbox.y < 0 || bbox.width < 10 || bbox.height < 10) return reply.code(400).send({ error: "Invalid region dimensions" });
    const fontSize = Math.round(Number(body.fontSize ?? metadata.font_size ?? 20));
    const lineSpacing = Number(body.lineSpacing ?? metadata.editor_line_spacing ?? 0.12);
    const alignment = String(body.alignment ?? metadata.editor_alignment ?? "center");
    if (fontSize < 8 || fontSize > 96 || lineSpacing < 0 || lineSpacing > 1 || !["left", "center", "right"].includes(alignment)) {
      return reply.code(400).send({ error: "Invalid typesetting controls" });
    }
    metadata.editor_font_size = fontSize;
    metadata.editor_line_spacing = lineSpacing;
    metadata.editor_alignment = alignment;
    database.prepare("UPDATE regions SET translation=?,bbox_json=?,metadata_json=?,status='EDITED' WHERE id=?")
      .run(translation.toUpperCase(), JSON.stringify(bbox), JSON.stringify(metadata), regionId);
    return { id: regionId, translation: translation.toUpperCase(), bbox, metadata, status: "EDITED" };
  });
  app.patch("/api/images/:imageId/crop", async (request, reply) => {
    const { imageId } = request.params as { imageId: string };
    const body = request.body as { cropTop?: number; cropBottom?: number; applyToAll?: boolean };
    const row = database.prepare(
      "SELECT i.id,i.height,i.output_path,j.id job_id,j.output_path job_output_path FROM images i JOIN jobs j ON j.id=i.job_id WHERE i.id=?"
    ).get(imageId) as SqlRow | undefined;
    if (!row) return reply.code(404).send({ error: "Image not found" });
    const height = Number(row.height) || 0;
    const cropTop = Math.round(Number(body?.cropTop ?? 0));
    const cropBottom = Math.round(Number(body?.cropBottom ?? 0));
    const invalid = cropError(height, cropTop, cropBottom);
    if (invalid) return reply.code(400).send({ error: invalid });
    const now = Date.now() / 1000;
    const targets = body?.applyToAll
      ? database.prepare("SELECT id,output_path FROM images WHERE job_id=?").all(String(row.job_id)) as SqlRow[]
      : [{ id: imageId, output_path: row.output_path }];
    const update = database.prepare("UPDATE images SET crop_top=?,crop_bottom=?,crop_updated_at=? WHERE id=?");
    for (const target of targets) {
      update.run(cropTop, cropBottom, now, String(target.id));
    }
    // The archive and the per-page download are both built from the rendered
    // PNGs, so every affected page's caches must go before the ZIP is rebuilt.
    for (const target of targets) {
      if (!target.output_path) continue;
      await clearPreviews(String(target.output_path), String(target.id));
      await clearDownloads(String(target.output_path));
    }
    const jobOutput = String(row.job_output_path ?? "");
    const archiveRebuilt = await rebuildArchive(jobOutput, request.log);
    return {
      imageId,
      cropTop,
      cropBottom,
      appliedTo: targets.length,
      archiveRebuilt,
    };
  });
  app.post("/api/images/:imageId/render", async (request, reply) => {
    const { imageId } = request.params as { imageId: string };
    const body = request.body as { regionId?: string };
    if (!body?.regionId) return reply.code(400).send({ error: "regionId is required" });
    const row = database.prepare(
      "SELECT r.*,i.output_path,i.metadata_path,j.input_path,j.output_path job_output_path,j.id job_id FROM regions r JOIN images i ON i.id=r.image_id JOIN jobs j ON j.id=i.job_id WHERE r.id=? AND r.image_id=?"
    ).get(body.regionId, imageId) as SqlRow | undefined;
    if (!row) return reply.code(404).send({ error: "Region not found" });
    const metadata = JSON.parse(String(row.metadata_json || "{}")) as Record<string, unknown>;
    if (metadata.reason === "REMOVAL_NOT_SAFE") return reply.code(409).send({ error: "Region removal is unsafe and must remain original" });
    const bbox = JSON.parse(String(row.bbox_json));
    const originalBBox = (metadata.bbox as object | undefined) ?? bbox;
    const inputPath = String(row.input_path);
    const cleaned = row.metadata_path
      ? resolve(String(row.metadata_path), "cleaned.png")
      : resolve(dirname(dirname(inputPath)), "metadata", "pipeline", "cleaned.png");
    const output = String(row.output_path);
    const color = Array.isArray(metadata.text_color) ? metadata.text_color : [15, 15, 15];
    const strokeColor = Array.isArray(metadata.stroke_fill) ? metadata.stroke_fill : color;
    try {
      await execFile(pythonPath, [
        reviewScript, "render", cleaned, output, output,
        "--bbox", JSON.stringify(bbox), "--original-bbox", JSON.stringify(originalBBox),
        "--translation", String(row.translation),
        "--font-size", String(metadata.editor_font_size ?? metadata.font_size ?? 20),
        "--line-spacing", String(metadata.editor_line_spacing ?? 0.12),
        "--alignment", String(metadata.editor_alignment ?? "center"), "--color", JSON.stringify(color),
        "--stroke-width", String(metadata.stroke_width ?? 0), "--stroke-color", JSON.stringify(strokeColor),
      ], { cwd: projectRoot, maxBuffer: 1024 * 1024 });
      const jobOutput = String(row.job_output_path);
      await rebuildArchive(jobOutput, request.log);
      database.prepare("UPDATE regions SET status='TYPESET' WHERE id=?").run(body.regionId);
      const review = database.prepare("SELECT count(*) count FROM regions r JOIN images i ON i.id=r.image_id WHERE i.job_id=? AND r.status='NEEDS_REVIEW'").get(String(row.job_id)) as SqlRow;
      const reviewCount = Number(review.count);
      database.prepare("UPDATE jobs SET review_count=?,status=CASE WHEN ?=0 AND status='REVIEW' THEN 'COMPLETED' ELSE status END,updated_at=? WHERE id=?")
      .run(reviewCount, reviewCount, Date.now() / 1000, String(row.job_id));
      await clearPreviews(output, imageId);
      return { imageId, regionId: body.regionId, status: "TYPESET", reviewCount };
    } catch (error) {
      request.log.warn(error);
      return reply.code(409).send({ error: stageErrorMessage(error) });
    }
  });
  app.get("/api/images/:imageId/download", async (request, reply) => {
    const { imageId } = request.params as { imageId: string };
    const row = database.prepare(
      "SELECT i.output_path,i.filename,i.crop_top,i.crop_bottom,i.crop_updated_at,j.options_json FROM images i JOIN jobs j ON j.id=i.job_id WHERE i.id=?"
    ).get(imageId) as SqlRow | undefined;
    if (!row) return reply.code(404).send({ error: "Image not found" });
    const source = String(row.output_path);
    // A page is delivered as .webp, or as a .zip when it is taller than the
    // archive split (the renderer decides, and prints the path it wrote). The
    // extension is therefore discovered, never assumed. A cached file older than
    // the rendered page — or older than the last crop — is stale, the same
    // freshness rule the preview uses.
    const { cropTop, cropBottom, cropUpdatedMs } = cropOf(row);
    const directory = resolve(dirname(source), "downloads");
    await mkdir(directory, { recursive: true });
    const stem = basename(source, extname(source));
    const candidates = [resolve(directory, `${stem}.webp`), resolve(directory, `${stem}.zip`)];
    let deliver: string | null = null;
    for (const candidate of candidates) {
      try {
        const sourceStat = await stat(source);
        const cachedStat = await stat(candidate);
        if (cachedStat.mtimeMs >= sourceStat.mtimeMs && cachedStat.mtimeMs >= cropUpdatedMs) deliver = candidate;
      } catch {
        // Not built yet, or removed: try the next candidate.
      }
    }
    if (!deliver) {
      await Promise.all(candidates.map((candidate) => rm(candidate, { force: true })));
      const { stdout } = await execFile(pythonPath, [reviewScript, "page", source, directory, "--crop-top", String(cropTop), "--crop-bottom", String(cropBottom)], { cwd: projectRoot });
      const written = String(stdout).trim().split("\n").filter(Boolean).at(-1);
      if (!written || !(await pathExists(written))) {
        return reply.code(409).send({ error: "Halaman belum punya hasil render." });
      }
      deliver = written;
    }
    const options = JSON.parse(String(row.options_json ?? "{}")) as Record<string, unknown>;
    const title = folderSlug(String(options.title_slug || options.comic_title || "komik"), "komik");
    const chapter = folderSlug(String(options.chapter || "1").replace(/^\s*(?:chapter|ch)\s*[-_. ]*/i, ""), "1");
    const page = folderSlug(String(row.filename).replace(/\.[^.]+$/, ""), "page");
    const suffix = extname(deliver).toLowerCase() === ".zip" ? ".zip" : ".webp";
    reply.header("Content-Disposition", `attachment; filename="${title}-ch-${chapter}-${page}${suffix}"`);
    reply.header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
    return reply.type(suffix === ".zip" ? "application/zip" : "image/webp").send(createReadStream(deliver));
  });
  app.get("/api/jobs/:jobId/download", async (request, reply) => {
    const { jobId } = request.params as { jobId: string };
    const row = database.prepare("SELECT status,output_path,options_json FROM jobs WHERE id=?").get(jobId) as SqlRow | undefined;
    if (!row) return reply.code(404).send({ error: "Job not found" });
    if (!['COMPLETED', 'REVIEW'].includes(String(row.status))) return reply.code(409).send({ error: "Output is not ready" });
    const output = String(row.output_path);
    reply.header("Content-Disposition", `attachment; filename="${jobDownloadFilename(row)}"`);
    reply.header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
    reply.header("Pragma", "no-cache");
    reply.header("Expires", "0");
    const suffix = extname(output).toLowerCase();
    const contentType = suffix === ".zip" ? "application/zip" : suffix === ".png" ? "image/png" : suffix === ".webp" ? "image/webp" : "image/jpeg";
    return reply.type(contentType).send(createReadStream(output));
  });
  app.get("/api/system/status", async () => {
    const counts = database.prepare("SELECT status,count(*) count FROM jobs GROUP BY status").all() as SqlRow[];
    const queue = Object.fromEntries(counts.map((row) => [String(row.status), Number(row.count)]));
    const processing = database.prepare("SELECT id,stage,heartbeat_at FROM jobs WHERE status='PROCESSING' LIMIT 1").get();
    return {
      cpu: { cores: cpus().length, loadAverage: loadavg() },
      memory: { total: totalmem(), available: freemem() },
      worker: { status: processing ? "processing" : "idle", job: processing ?? null },
      queue,
    };
  });
  app.get("/api/settings", async () => database.prepare("SELECT key,value,updated_at FROM settings ORDER BY key").all());
  app.patch("/api/settings", async (request, reply) => {
    const body = request.body as Record<string, string | number>;
    const numeric = new Set(["minimum_free_ram_mb", "poll_interval_seconds"]);
    const text = new Set(["ninerouter_model", "ninerouter_fallback_models"]);
    const now = Date.now() / 1000;
    for (const [key, raw] of Object.entries(body || {})) {
      if (!numeric.has(key) && !text.has(key)) return reply.code(400).send({ error: `Setting is not editable: ${key}` });
      if (text.has(key)) {
        const value = String(raw).trim();
        if (!value || value.length > 500 || !/^[A-Za-z0-9._,\/-]+$/.test(value)) return reply.code(400).send({ error: `Invalid value for ${key}` });
        database.prepare("INSERT INTO settings(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at").run(key, value, now);
        continue;
      }
      const value = Number(raw);
      if (!Number.isFinite(value) || value <= 0) return reply.code(400).send({ error: `Invalid value for ${key}` });
      database.prepare("INSERT INTO settings(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at").run(key, String(value), now);
    }
    return database.prepare("SELECT key,value,updated_at FROM settings ORDER BY key").all();
  });
  app.post("/api/settings/test-translator", async (request, reply) => {
    const model = String((request.body as { model?: string })?.model || "").trim();
    if (!model || model.length > 120 || !/^[A-Za-z0-9._\/-]+$/.test(model)) return reply.code(400).send({ error: "Model tidak valid" });
    const envText = await readFile(resolve(projectRoot, ".env.9router"), "utf8");
    const values = Object.fromEntries(envText.split(/\r?\n/).filter(line => line && !line.startsWith("#") && line.includes("=")).map(line => { const at=line.indexOf("="); return [line.slice(0,at),line.slice(at+1)]; }));
    const base = String(values.NINEROUTER_BASE_URL || "http://127.0.0.1:20128/v1").replace(/\/$/, "");
    const response = await fetch(`${base}/chat/completions`, { method:"POST", headers:{"Content-Type":"application/json", ...(values.NINEROUTER_API_KEY?{Authorization:`Bearer ${values.NINEROUTER_API_KEY}`}:{})}, body:JSON.stringify({model,messages:[{role:"user",content:'Balas tepat dengan JSON: {"ok":true}'}],temperature:0,response_format:{type:"json_object"}}), signal:AbortSignal.timeout(45000) });
    const payload = await response.text();
    if (!response.ok) return reply.code(502).send({error:`9router ${response.status}: ${payload.slice(0,300)}`});
    return {ok:true,model};
  });

  return app;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const app = await buildApp();
  await app.listen({ host: process.env.TRANSLATOR_API_HOST || "127.0.0.1", port: Number(process.env.TRANSLATOR_API_PORT || 3210) });
}
