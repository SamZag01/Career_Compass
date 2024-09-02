from flask import Flask
from flask_socketio import SocketIO,emit

app = Flask(__name__)

from app import views
