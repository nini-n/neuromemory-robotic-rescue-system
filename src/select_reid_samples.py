import zipfile
import shutil
from pathlib import Path
from collections import defaultdict


# ============================================================
# Market-1501 ZIP Sample Selector
# Automatically selects a small subset for NeuroMemory validation.
# ============================================================

ZIP_PATH = Path("data/archive.zip")

EXTRACT_DIR = Path("data/market1501_extracted")
OUTPUT_DIR = Path("data/sample_reid")

NUM_IDENTITIES = 10
IMAGES_PER_IDENTITY = 4


def get_person_id(filename: str) -> str:
    """
    Market-1501 filenames usually start with identity ID:
    0002_c1s1_000451_03.jpg -> 0002
    """
    return Path(filename).name.split("_")[0]


def extract_zip_if_needed():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"ZIP file not found: {ZIP_PATH}")

    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.rglob("*.jpg")):
        print(f"Dataset already extracted: {EXTRACT_DIR}")
        return

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Extracting ZIP file:\n{ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    print(f"Extracted to: {EXTRACT_DIR.resolve()}")


def find_source_folder() -> Path:
    preferred_names = [
        "bounding_box_test",
        "bounding_box_train",
        "query",
    ]

    for preferred in preferred_names:
        matches = [p for p in EXTRACT_DIR.rglob(preferred) if p.is_dir()]
        if matches:
            print(f"Using source folder: {matches[0]}")
            return matches[0]

    raise FileNotFoundError(
        "Could not find bounding_box_test, bounding_box_train, or query inside extracted dataset."
    )


def collect_images(source_dir: Path):
    grouped = defaultdict(list)

    for file in source_dir.iterdir():
        if not file.is_file():
            continue

        if not file.name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        person_id = get_person_id(file.name)

        # Market-1501 uses -1 for junk/background images.
        if person_id in ["-1", "0000"]:
            continue
        
        grouped[person_id].append(file)

    return grouped


def main():
    extract_zip_if_needed()

    source_dir = find_source_folder()
    grouped = collect_images(source_dir)

    valid_ids = {
        pid: files
        for pid, files in grouped.items()
        if len(files) >= IMAGES_PER_IDENTITY
    }

    if len(valid_ids) < NUM_IDENTITIES:
        raise ValueError(
            f"Not enough identities with at least {IMAGES_PER_IDENTITY} images. "
            f"Found only {len(valid_ids)} valid identities."
        )

    selected_ids = sorted(valid_ids.keys())[:NUM_IDENTITIES]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean old files
    for old_file in OUTPUT_DIR.glob("*"):
        if old_file.is_file():
            old_file.unlink()

    print("\nSelected identities:")
    for pid in selected_ids:
        selected_files = sorted(valid_ids[pid])[:IMAGES_PER_IDENTITY]
        print(f"  ID {pid}: {len(selected_files)} images")

        for idx, src in enumerate(selected_files, start=1):
            new_name = f"id_{pid}_sample_{idx}_{src.name}"
            dst = OUTPUT_DIR / new_name
            shutil.copy2(src, dst)

    print(f"\nDone. Selected images copied to:")
    print(OUTPUT_DIR.resolve())

    print("\nSelected files:")
    for file in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {file.name}")


if __name__ == "__main__":
    main()