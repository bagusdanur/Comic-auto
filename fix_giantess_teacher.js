const { DatabaseSync } = require("node:sqlite");
const db = new DatabaseSync("data/translator.sqlite3");
const jobId = "e1d6e67e-2310-4817-a4b7-2215ff92e167";
const row = db.prepare(`
  SELECT r.id, r.image_id, r.translation, r.bbox_json, r.metadata_json,
         i.output_path, i.metadata_path, j.output_path AS job_output
  FROM regions r
  JOIN images i ON i.id = r.image_id
  JOIN jobs j ON j.id = i.job_id
  WHERE r.translation LIKE '%Pak Guru%'
    AND (j.id = ? OR j.input_path LIKE '%giantess-buddy%chapter-2%')
`).get(jobId);
if (!row) throw new Error("Region Pak Guru tidak ditemukan");
db.prepare("UPDATE regions SET translation=?, status='EDITED' WHERE id=?")
  .run("BU GURU, ADA YANG NEMPEL DI PUNGGUNG IBU!", row.id);
console.log(JSON.stringify(row));
