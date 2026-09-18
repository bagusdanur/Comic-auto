import shutil
import sqlite3
from pathlib import Path

comic_id = "19b172ec-1dde-4548-a163-e3da8d19ba82"
root = Path("/home/ryukomik/ryukomik-auto-comic-translator").resolve()
database = sqlite3.connect(root / "data/translator.sqlite3")
row = database.execute("SELECT slug FROM comics WHERE id=?", (comic_id,)).fetchone()
if row:
    slug = row[0]
    work_paths = [Path(item[0]).resolve() for item in database.execute("SELECT work_path FROM memory_jobs WHERE comic_id=?", (comic_id,))]
    database.execute("DELETE FROM comics WHERE id=?", (comic_id,))
    database.commit()
    targets = work_paths + [root / "data/memory" / slug, root / "data/memory" / f"{slug}.json"]
    allowed = [(root / "data/memory-analysis").resolve(), (root / "data/memory").resolve()]
    for target in targets:
        target = target.resolve()
        if not any(target != base and base in target.parents for base in allowed):
            raise RuntimeError(f"Unsafe cleanup target: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
database.close()

