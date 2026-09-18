"""Align two user-approved generic terms; retain every other title entry."""
import json,shutil,sqlite3
from pathlib import Path
c=sqlite3.connect('data/translator.sqlite3')
assert c.execute("select count(*) from jobs where status='PROCESSING'").fetchone()[0]==0,'Wait for active jobs before changing memory'
p=Path('data/memory/erica-when-a-friend-is-obsessed-with-me.json')
backup=Path('artifacts/erica-memory-before-natural-v2.json')
assert not backup.exists(),'Already applied or backup exists'
shutil.copy2(p,backup)
d=json.loads(p.read_text())
d.update({'S-Class Mercenary':'tentara bayaran kelas S','Merchant Guild':'serikat pedagang'})
temporary=p.with_suffix('.natural-v2.tmp');temporary.write_text(json.dumps(d,ensure_ascii=False,indent=2));temporary.replace(p)
print('Aligned 2 generic terms; retained all other title memory entries')
