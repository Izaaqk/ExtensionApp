from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
from urllib.parse import urlparse

app = Flask(__name__)

@app.route("/predict")
def hello():
    return "Estará prediciendo en poco tiempo"
