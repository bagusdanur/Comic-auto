const { DatabaseSync } = require("node:sqlite");
const db = new DatabaseSync("data/translator.sqlite3", { readOnly: true });
for (const row of db.prepare("SELECT * FROM jobs WHERE id IN (?,?,?,?,?,?,?,?)").all(
  "841415f5-0117-45f1-ab6c-2e7dc6ecb46a", "9985fe33-fe2a-471d-a42a-7ca0f2c81cdb",
  "f169aa08-cc23-4dcb-aeaf-521ca98d7e8a", "4bdb189b-18f0-4ca8-974d-c2ff6f03c939",
  "f330e0c6-c463-4b3a-a48c-10346bf79eee", "08a1de9f-8b57-42df-92e7-75d74082ca09",
  "92b13ca7-93cc-439c-816c-df0fbab69214", "c3e278e8-1998-4796-9209-1425f012dfb1"
)) console.log(JSON.stringify(row));
