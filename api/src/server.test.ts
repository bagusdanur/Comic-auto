import assert from "node:assert/strict";
import test from "node:test";
import { buildApp, jobDownloadFilename } from "./server.js";

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
