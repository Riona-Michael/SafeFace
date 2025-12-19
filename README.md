# SafeFace
AI-powered tool to detect deepfake images by classifying faces as real or fake.


0)Environment & install
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install tensorflow mtcnn opencv-python matplotlib numpy pandas scikit-learn pillow


2)Run it (from project root):
python code/preprocess.py
Check: processed images appear under dataset/processed/real and dataset/processed/fake. If many “Saved:” prints — good.


2)Run training:
python code/train.py
Expect logs and saved files: results/best_model.h5, results/final_model.h5, results/metrics.json.


3)Place a test image at sample_test.jpg at project root


4)Run:
python app.py
upload images and click analyze too see the prediction !

