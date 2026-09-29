import assert from "node:assert/strict";
import test from "node:test";
import { buildApp, jobDownloadFilename, stageErrorMessage } from "./server.js";

test("stage error message surfaces the pipeline reason, not the command", () => {
  // What execFile actually rejects with: the command line, a Python traceback
  // and the real ValueError at the end.
  const failure = new Error([
    "Command failed: /home/ryukomik/ryukomik-auto-comic-translator/.venv/bin/python review_render.py render a.png b.png c.png --bbox {...}",
    "Traceback (most recent call last):",
    '  File "review_render.py", line 73, in render',
    "    raise ValueError(",
    "ValueError: Teks tidak muat di kotak 60x24 bahkan pada font 17px. Perbesar kotak (Width/Height) atau pendekkan terjemahannya.",
    "",
  ].join("\n"));
  assert.equal(
    stageErrorMessage(failure),
    "Teks tidak muat di kotak 60x24 bahkan pada font 17px. Perbesar kotak (Width/Height) atau pendekkan terjemahannya.",
  );
  // A traceback without a recognisable exception must not leak the command line.
  assert.equal(stageErrorMessage(new Error("Command failed: python x.py")), "Render gagal dijalankan.");
  assert.equal(stageErrorMessage(new Error("")), "Render gagal.");
  assert.equal(stageErrorMessage(undefined), "Render gagal.");
});

test("download filename uses comic title and chapter", () => {
  assert.equal(jobDownloadFilename({
    output_path: "/tmp/chapter-translated.zip",
    options_json: JSON.stringify({ comic_title: "You Like Someone With That Face?", chapter: "Chapter 29" }),
  }), "you-like-someone-with-that-face-ch-29.zip");
  assert.equal(jobDownloadFilename({
    output_path: "/tmp/result.png",
    options_json: JSON.stringify({ title_slug: "judul-komik", chapter: "ch-4" }),
  }), "judul-komik-ch-4.png");
});

test("health-style system endpoint and settings are available", async () => {
  const app = await buildApp();
  const status = await app.inject({ method: "GET", url: "/api/system/status" });
  assert.equal(status.statusCode, 200);
  assert.equal(typeof status.json().memory.available, "number");
  const settings = await app.inject({ method: "GET", url: "/api/settings" });
  assert.equal(settings.statusCode, 200);
  assert.ok(Array.isArray(settings.json()));
  await app.close();
});

test("missing jobs return 404 and render validates its region", async () => {
  const app = await buildApp();
  assert.equal((await app.inject({ method: "GET", url: "/api/jobs/missing" })).statusCode, 404);
  assert.equal((await app.inject({ method: "DELETE", url: "/api/jobs/missing" })).statusCode, 404);
  assert.equal((await app.inject({ method: "POST", url: "/api/images/missing/render" })).statusCode, 400);
  assert.equal((await app.inject({ method: "POST", url: "/api/images/missing/render", payload: { regionId: "missing" } })).statusCode, 404);
  await app.close();
});

test("comic memory endpoints are isolated and validate missing comics", async () => {
  const app = await buildApp();
  assert.equal((await app.inject({ method: "GET", url: "/api/comics/missing/memory" })).statusCode, 404);
  assert.equal((await app.inject({ method: "POST", url: "/api/comics/missing/memory/analyze" })).statusCode, 404);
  assert.equal((await app.inject({ method: "DELETE", url: "/api/comics/missing/memory/references/missing" })).statusCode, 404);
  await app.close();
});
