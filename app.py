"""
This is the main application file for the weather miner application.
"""

from flask import Flask

weather_miner = Flask(__name__)

@weather_miner.route("/")
def index():
    """
    This is the index page.
    """
    return "<p>Coming Soon: A machine learning application to mine weather patterns in history!</p>"

if __name__ == "__main__":
    weather_miner.run(
        # host="0.0.0.0",
        port=int(3000),
        debug=True
    )
