from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from pptx import Presentation


PPTX = Path(r"C:\Users\user\Downloads\PRESENTATION SLIDES 2026-MSC JOOUST.pptx")
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}


def shape_text(shape):
    if not hasattr(shape, "text"):
        return ""
    return shape.text.strip()


def extract_comments():
    comments = []
    with ZipFile(PPTX) as zf:
        for name in sorted(zf.namelist()):
            if not re.match(r"ppt/comments/comment\d+\.xml$", name):
                continue
            root = ET.fromstring(zf.read(name))
            slide_num = int(re.search(r"comment(\d+)\.xml$", name).group(1))
            texts = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    texts.append(elem.text.strip())
            if texts:
                comments.append((slide_num, texts))
    return comments


def main():
    prs = Presentation(PPTX)
    print(f"slides={len(prs.slides)}")
    for idx, slide in enumerate(prs.slides, start=1):
        lines = []
        for shape in slide.shapes:
            text = shape_text(shape)
            if text:
                lines.extend([ln.strip() for ln in text.splitlines() if ln.strip()])
        print(f"\n--- SLIDE {idx} ---")
        for line in lines[:30]:
            print(line)
    print("\n=== COMMENTS ===")
    comments = extract_comments()
    if not comments:
        print("NO_COMMENTS_FOUND")
    for slide_num, texts in comments:
        print(f"\n--- COMMENT PART {slide_num} ---")
        for text in texts:
            print(text)


if __name__ == "__main__":
    main()
