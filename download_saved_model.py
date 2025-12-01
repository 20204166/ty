import shutil
from pathlib import Path

import kagglehub


def main():
    # 1) Download dataset from Kaggle using kagglehub
    print("Downloading dataset from Kaggle with kagglehub...")
    src_path = kagglehub.dataset_download("bekithembancube/saved-model")
    src_dir = Path(src_path)
    print(f"Downloaded dataset to: {src_dir}")

    # 2) Target dir inside your repo: app/models/saved_model
    dest_dir = Path("app/models/saved_model")
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Copying files into: {dest_dir.resolve()}")

    # 3) Copy everything over (like cp /kaggle/input/saved-model/* app/models/saved_model/)
    for item in src_dir.iterdir():
        target = dest_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    print("\nFiles in app/models/saved_model:")
    for p in sorted(dest_dir.rglob("*")):
        if p.is_file():
            print(" -", p.relative_to(dest_dir))


if __name__ == "__main__":
    main()
