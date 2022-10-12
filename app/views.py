import os

from app import app, model
from app.utils import (
    allowed_file,
    load_and_transform_image
)
from werkzeug.utils import secure_filename
from flask import (
    request,
    redirect,
    render_template,
    abort,
    flash,
    url_for
)


@app.route("/")
def index():
    return render_template("index.html")


@app.errorhandler(404)
def invalid_route(e):
    return render_template("page_not_found.html"), 404


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("upload.html")

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            abort(404)

        file = request.files["file"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return redirect(url_for("predict", filename=filename))
        else:
            abort(404)
    else:
        abort(404)


@app.route("/predict/<string:filename>", methods=["GET", "POST"])
def predict(filename):
    file = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(file):
        print('No such file:', filename)
        abort(404)
    else:
        img = load_and_transform_image(file)
        label, confidence = model.predict(img)
        
        if label and confidence:
            label = " ".join(label.split('_'))
            return render_template("prediction.html", img_src= "../static/images/" + filename, label=label, confidence=confidence)
        else:
            abort(404)