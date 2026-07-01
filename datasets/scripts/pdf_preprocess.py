from pathlib import Path
from pdf2image import convert_from_path
import cv2
import numpy as np
import os

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "raw"
OUTPUT_DIR = ROOT / "processed" / "enhanced_images"

POPPLER_PATH = None
if os.name == "nt":
    POPPLER_PATH = r"D:\Poppler\poppler-26.02.0\Library\bin"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pdf_files = list(RAW_DIR.rglob("*.pdf"))

if not pdf_files:
    print("No PDF Found")
    exit()

print(f"\nFound {len(pdf_files)} PDF(s)\n")

for idx, pdf_file in enumerate(pdf_files, start=1):
    relative = pdf_file.relative_to(RAW_DIR)
    save_folder = OUTPUT_DIR / relative.parent / pdf_file.stem
    save_folder.mkdir(parents=True, exist_ok=True)

    print(f"[{idx}/{len(pdf_files)}] {pdf_file.name}")

    try:
        if POPPLER_PATH:
            pages = convert_from_path(
                str(pdf_file),
                dpi=350,
                thread_count=4,
                poppler_path=POPPLER_PATH
            )
        else:
            pages = convert_from_path(
                str(pdf_file),
                dpi=350,
                thread_count=4
            )

        for i, page in enumerate(pages, start=1):
            image = np.array(page)

            if len(image.shape) == 3 and image.shape[2] == 4:image = image[:, :, :3]

            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray = cv2.fastNlMeansDenoising(gray)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            thresh = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                15
            )

            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])

            sharp = cv2.filter2D(thresh, -1, kernel)

            if np.mean(sharp) > 250:
                print(f"Skipped blank page {i}")
                continue
            output_file = save_folder / f"page_{i}.png"

            success = cv2.imwrite(
                str(output_file),
                sharp,
                [cv2.IMWRITE_PNG_COMPRESSION, 3]
            )

            if not success:
                print(f"Failed to save: {output_file}")

            print(
                f"Saved {len(list(save_folder.glob('*.png')))} page(s) -> {save_folder}"
            )

    except Exception as e:
        print(f"Error processing {pdf_file.name}: {e}")

print("\nFinished.")