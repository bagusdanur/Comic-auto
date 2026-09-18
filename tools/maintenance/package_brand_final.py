import zipfile
from pathlib import Path

root = Path("/tmp/ch29-brand-final")
with zipfile.ZipFile("/tmp/chapter-translated-v5-brand-final.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for page in sorted(root.glob("[0-9][0-9][0-9].png")):
        archive.write(page, page.name)

