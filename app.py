from flask import Flask, render_template, request
from main import search_for_stock, stock_prediction

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("layout.html")


@app.route('/', methods=["POST"])
def request_stock():
    """

    """
    stock_name = request.form['sname']
    stock_name = stock_name.upper()
    print(stock_name)
    if search_for_stock(stock_name):
        display(stock_name)
    else:
        predict(stock_name)


def display(symbol):
    """

    """
    data = []
    with open('archive/stocks/'+symbol+'/'+symbol+'.txt', 'r') as handler:
        for item in handler:
            curr_spot = item[:-1]
            data.append(curr_spot)
    return render_template("stocks_details.html")


def predict(symbol):
    """

    """
    data = predict(symbol)
    return render_template("stocks_details.html")


if __name__ == "__main__":
    app.run()