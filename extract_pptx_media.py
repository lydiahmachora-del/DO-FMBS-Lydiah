from pathlib import Path
from zipfile import ZipFile
import shutil


PPTX = Path(r"C:\Users\user\Downloads\PRESENTATION SLIDES 2026-MSC JOOUST.pptx")
OUT = Path("extracted_pptx_media")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    with ZipFile(PPTX) as zf:
        for name in zf.namelist():
            if name.startswith("ppt/media/"):
                target = OUT / Path(name).name
                target.write_bytes(zf.read(name))
                print(target)


if __name__ == "__main__":
    main()
