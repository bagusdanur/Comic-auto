import zipfile
from pathlib import Path

root = Path("/tmp/ch29-font-final")
with zipfile.ZipFile("/tmp/chapter-translated-v4-font-final.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for page in sorted(root.glob("[0-9][0-9][0-9].png")):
        archive.write(page, page.name)

