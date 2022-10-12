from app.settings import ALLOWED_EXTENSIONS

from PIL import Image
from numpy import expand_dims


def allowed_file(filename):
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_and_transform_image(path):
    image = Image.open(path)
    image = expand_dims(image, -1)
    
    return image