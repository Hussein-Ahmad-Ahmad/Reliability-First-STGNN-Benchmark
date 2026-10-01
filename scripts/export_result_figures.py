"""Export the original embedded figure pixels and a source index from a PDF.

Requires PyMuPDF for this optional publishing step, not for release verification.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    import fitz

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    document = fitz.open(args.pdf)
    destination = ROOT / "figures" / "results"
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for page_number, page in enumerate(document, 1):
        captions = [block for block in page.get_text("blocks")
                    if re.match(r"FIGURE (?:A\.)?\d+:", block[4])]
        if not captions:
            continue
        counters = {}
        for info in page.get_image_info(xrefs=True):
            # Publisher headers are small repeated images, not scientific figures.
            if info["width"] < 400 or info["height"] < 150:
                continue
            box = info["bbox"]
            below = [c for c in captions if c[1] >= box[3] - 3]
            if not below:
                raise ValueError(f"No caption below image on page {page_number}")
            caption = min(below, key=lambda c: c[1] - box[3])
            label = re.match(r"FIGURE ((?:A\.)?\d+):", caption[4]).group(1)
            counters[label] = counters.get(label, 0) + 1
            name = f"figure_{label.replace('.', '_')}_{counters[label]}.png"
            pixmap = fitz.Pixmap(document, info["xref"])
            image_entry = next(x for x in page.get_images(full=True)
                               if x[0] == info["xref"])
            if image_entry[1]:
                pixmap = fitz.Pixmap(pixmap, fitz.Pixmap(document, image_entry[1]))
            path = destination / name
            pixmap.save(path)
            entries.append({"figure": label, "page": page_number,
                            "path": path.relative_to(ROOT).as_posix(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "caption": " ".join(caption[4].split())})
    expected = {str(i) for i in range(1, 11)} | {f"A.{i}" for i in range(1, 14)}
    if {entry["figure"] for entry in entries} != expected:
        raise ValueError("Figure coverage differs from the expected 10 + 13 figures")
    output = ROOT / "results" / "release" / "figure_index.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump({"source_pdf_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
                   "source_pages": len(document),
                   "export": "Original embedded pixels; no redrawing or numerical transformation",
                   "figures": entries}, handle, indent=2)
        handle.write("\n")
    print(f"Exported {len(entries)} figure panels covering {len(expected)} figures")


if __name__ == "__main__":
    main()
