from paddleocr import PaddleOCR
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = ROOT / "processed" / "enhanced_images"
OCR_ROOT = ROOT / "processed" / "ocr_text"

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)

image_folders = [
    p for p in IMAGE_ROOT.rglob("*")
    if p.is_dir()
]

for folder in image_folders:

    pngs = list(folder.glob("*.png"))

    if not pngs:
        continue

    relative = folder.relative_to(IMAGE_ROOT)

    output_txt = OCR_ROOT / relative.with_suffix(".txt")

    output_txt.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nOCR: {folder}")

    all_text = []

    for image in sorted(pngs):

        result = ocr.ocr(str(image), cls=True)

        if result and result[0]:

            for line in result[0]:

                try:

                    text = line[1][0]
                    confidence = line[1][1]

                    if confidence >= 0.70:
                        all_text.append(text)

                except:
                    pass

    output_txt.write_text(
        "\n".join(all_text),
        encoding="utf-8"
    )

    print(f"Saved: {output_txt}")

print("\nOCR Complete")