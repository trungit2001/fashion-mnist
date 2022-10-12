import os

from flask import Flask
from app.settings import (
    UPLOAD_FOLDER, 
    IMAGE_SIZE, 
    MODEL_WEIGHTS_PATH
)
from app.models.model import FashionMNIST


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = FashionMNIST(input_shape=IMAGE_SIZE)
model.load_weights(os.path.join(MODEL_WEIGHTS_PATH, "fashion_mnist.h5"))


from app.views import *

