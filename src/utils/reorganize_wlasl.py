import os
import json
import shutil
from tqdm import tqdm
import argparse

def main(subset: int):
    # Load JSON
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(DST_DIR, exist_ok=True)

    print(f"Organizing WLASL{subset} ...\n")

    count = 0
    for entry in tqdm(data):
        gloss = entry["gloss"]
        instances = entry["instances"]

        # Tạo folder cho label
        gloss_dir = os.path.join(DST_DIR, gloss)
        os.makedirs(gloss_dir, exist_ok=True)

        for inst in instances:
            vid = inst["video_id"]
            src = os.path.join(SRC_VIDEO_DIR, f"{vid}.mp4")
            dst = os.path.join(gloss_dir, f"{vid}.mp4")

            if not os.path.exists(src):
                print(f"[WARNING] Missing video: {src}")
                continue

            shutil.copy2(src, dst)  # copy2 giữ metadata (ctime, mtime,...)
        
        count += 1
        if count == subset:
            break

    print("\nDone! Videos have been structured by gloss.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="100")
    args = parser.parse_args()
    subset = int(args.subset)

    JSON_PATH = "data/wlasl/WLASL_v0.3.json"
    SRC_VIDEO_DIR = "data/wlasl/videos"
    DST_DIR = f"data/wlasl/wlasl{str(subset)}"

    main(subset)
