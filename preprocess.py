# code/preprocess.py
import os
from mtcnn import MTCNN
from PIL import Image
import numpy as np

INPUT_DIR = "./dataset_raw"
OUTPUT_DIR = "./dataset/processed"
TARGET_SIZE = (128, 128)

detector = MTCNN()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def crop_face_save(in_path, out_path, target_size=TARGET_SIZE):
    try:
        img = Image.open(in_path).convert("RGB")
        arr = np.asarray(img)
        results = detector.detect_faces(arr)
        if not results:
            # fallback: center resize
            img = img.resize(target_size)
            img.save(out_path)
            return True
        face = max(results, key=lambda r: r['box'][2]*r['box'][3])
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        crop = arr[y:y+h, x:x+w]
        pil = Image.fromarray(crop).resize(target_size)
        pil.save(out_path)
        return True
    except Exception as e:
        print("ERROR:", in_path, e)
        return False

def process_class(class_name):
    src = os.path.join(INPUT_DIR, class_name)
    dst = os.path.join(OUTPUT_DIR, class_name)
    ensure_dir(dst)
    for fname in os.listdir(src):
        fin = os.path.join(src, fname)
        fout = os.path.join(dst, fname)
        success = crop_face_save(fin, fout)
        if success:
            print("Saved:", fout)

def main():
    ensure_dir(OUTPUT_DIR)
    for c in ["real", "fake"]:
        if os.path.exists(os.path.join(INPUT_DIR, c)):
            process_class(c)
        else:
            print("Warning: folder not found:", os.path.join(INPUT_DIR, c))

if __name__ == "__main__":
    main()
