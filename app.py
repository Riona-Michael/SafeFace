from flask import Flask, render_template, request
import os
from mycode.inference import predict, MODEL_PATH


UPLOAD_FOLDER = "static/uploads"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            label, prob, _ = predict(image_path)

            result = {
                "label": label,
                "prob": f"{prob:.2f}"
            }

    return render_template("index.html",
                           result=result,
                           image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
