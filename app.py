from flask import Flask

from markupsafe import escape

app = Flask(__name__)

@app.route("/")
def weather_miner():
    return "<p>A machine learning application to mine weather patterns in history!</p>"
