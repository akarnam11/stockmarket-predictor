from flask import Flask, render_template, request
from main import search, stock_prediction

app = Flask(__name__)
