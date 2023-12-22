import os

from flask import Flask
from app import app as aplication

if __name__ == "__main__":
    host = os.environ.get('FLASK_HOST', "0.0.0.0")
    port = os.environ.get('FLASK_PORT', '80')
    aplication.run(host=host, port=port)