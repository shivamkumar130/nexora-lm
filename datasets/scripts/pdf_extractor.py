from pathlib import Path

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from pypdf import PdfReader

# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "raw"
OUTPUT_DIR = ROOT / "processed" / "clean_text"
IMAGE_ROOT = ROOT / "processed" / "enhanced_images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# OCR MODEL
# ==================================================

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    show_log=False,
)

# ==================================================
# FIND PDF FILES
# ==================================================

pdf_files = list(RAW_DIR.rglob("*.pdf"))

if not pdf_files:
    print("No PDF Found")
    raise SystemExit

print(f"\nFound {len(pdf_files)} PDFs")

# ==================================================
# PROCESS PDFS
# ==================================================

for pdf_file in pdf_files:

    relative = pdf_file.relative_to(RAW_DIR)

    output_file = OUTPUT_DIR / relative.with_suffix(".txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Processing: {pdf_file.name}")
    print("=" * 60)

    extracted_text = ""

    try:

        # ==================================================
        # TRY DIGITAL EXTRACTION
        # ==================================================

        with open(pdf_file, "rb") as f:

            reader = PdfReader(f)

            for page_num, page in enumerate(reader.pages, start=1):

                page_text = page.extract_text()

                if page_text:

                    page_text = page_text.replace("\x00", "").strip()

                    if page_text:
                        extracted_text += f"\n===== PAGE {page_num} =====\n"
                        extracted_text += page_text + "\n"

        # ==================================================
        # DIGITAL PDF
        # ==================================================

        if len(extracted_text.split()) > 100:

            output_file.write_text(
                extracted_text.strip(),
                encoding="utf-8",
            )

            print("✓ Digital PDF detected")
            print(f"✓ Saved: {output_file}")

        # ==================================================
        # SCANNED PDF
        # ==================================================

        else:

            print("✓ Scanned PDF detected")

            image_folder = (
                IMAGE_ROOT
                / relative.parent
                / pdf_file.stem
            )

            images = sorted(image_folder.glob("*.png"))

            if not images:
                print("✗ No preprocessed images found.")
                continue

            ocr_lines = []

            for i, image_path in enumerate(images, start=1):

                print(
                    f"[{pdf_file.stem}] OCR Page {i}/{len(images)}"
                )

                image = np.array(
                    Image.open(image_path).convert("RGB")
                )

                result = ocr.ocr(image, cls=True)

                ocr_lines.append(f"===== PAGE {i} =====")

                if result and result[0]:

                    for line in result[0]:

                        try:

                            text = line[1][0]
                            confidence = float(line[1][1])

                            if confidence >= 0.80:

                                text = text.replace("\x00", "").strip()
                                text = " ".join(text.split())
                                text = text.replace("|", "I")

                                if text:
                                    ocr_lines.append(text)

                        except Exception:
                            continue

            ocr_text = "\n".join(ocr_lines)

            if len(ocr_text.split()) > 100:

                output_file.write_text(
                    ocr_text.strip(),
                    encoding="utf-8",
                )

                print(f"✓ Saved: {output_file}")

            else:
                print("✗ OCR produced very little text")

    except Exception as e:

        print("=" * 60)
        print(f"ERROR : {pdf_file.name}")
        print(type(e).__name__)
        print(e)
        print("=" * 60)

print("\n" + "=" * 60)
print("PDF EXTRACTION COMPLETED")
print(f"Processed {len(pdf_files)} PDF(s)")
print("=" * 60)