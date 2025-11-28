SafeFace Prototype - How to Run

1) Preprocess Images:
   python preprocess.py
   Input: dataset_raw/real, dataset_raw/fake
   Output: dataset/processed/real, dataset/processed/fake

2) Train Model:
   python train.py
   Output: results/best_model.h5, results/metrics.json

3) Inference:
   python inference.py
   Input: sample_test.jpg
   Output: results/sample_prediction.png

Requirements:
- Python 3.x
- tensorflow, mtcnn, opencv-python, numpy, pandas, pillow, matplotlib, scikit-learn
