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
    const row = database.prepare("SELECT output_path FROM images WHERE id=?").get(imageId) as SqlRow | undefined;
    if (!row) return reply.code(404).send({ error: "Image not found" });
    const source = String(row.output_path);
    // Webtoon review shows every page stacked at full width, so the reader can
    // ask for a wider render than the 420 px single-page default. The width is
    // part of the cache filename; otherwise a wide request would poison the
    // cached thumbnail every other view relies on.
    const requested = Number.parseInt(String((request.query as Record<string, string>).w ?? ""), 10);
    const maxWidth = Number.isFinite(requested) ? Math.min(1600, Math.max(120, requested)) : 420;
    const destination = resolve(dirname(source), "previews", `${imageId}@${maxWidth}.jpg`);
    let refreshPreview = false;
    try {
      await access(destination);
      // Regenerate when the rendered page is newer than the cached preview.
      const [sourceStat, previewStat] = await Promise.all([stat(source), stat(destination)]);
      refreshPreview = sourceStat.mtimeMs > previewStat.mtimeMs;
    } catch {
      refreshPreview = true;
    }
    if (refreshPreview) {
      await mkdir(dirname(destination), { recursive: true });
      await execFile(pythonPath, [reviewScript, "preview", source, destination, "--max-width", String(maxWidth)], { cwd: projectRoot });
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
      if (extname(jobOutput).toLowerCase() === ".zip") {
        await execFile(pythonPath, [reviewScript, "archive", dirname(output), jobOutput], { cwd: projectRoot });
      }
      database.prepare("UPDATE regions SET status='TYPESET' WHERE id=?").run(body.regionId);
      const review = database.prepare("SELECT count(*) count FROM regions r JOIN images i ON i.id=r.image_id WHERE i.job_id=? AND r.status='NEEDS_REVIEW'").get(String(row.job_id)) as SqlRow;
      const reviewCount = Number(review.count);
      database.prepare("UPDATE jobs SET review_count=?,status=CASE WHEN ?=0 AND status='REVIEW' THEN 'COMPLETED' ELSE status END,updated_at=? WHERE id=?")
      .run(reviewCount, reviewCount, Date.now() / 1000, String(row.job_id));
      // Previews are cached per requested width (`{imageId}@{width}.jpg`), so a
      // re-render must invalidate every width variant, not just one.
      const previewDir = resolve(dirname(output), "previews");
      try {
        for (const name of await readdir(previewDir)) {
          if (name.startsWith(`${imageId}@`) || name === `${imageId}.jpg`) {
            await rm(resolve(previewDir, name), { force: true });
          }
        }
      } catch {
        // No previews directory yet: nothing to invalidate.
      }
      return { imageId, regionId: body.regionId, status: "TYPESET", reviewCount };
    } catch (error) {
      request.log.warn(error);
      return reply.code(409).send({ error: error instanceof Error ? error.message : "Render failed" });
    }
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
