import flask
import json
import numpy as np
import operator
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
#import urllib2

app = flask.Flask(__name__)

#CatBoost
model=CatBoostClassifier(iterations=20, depth=7, learning_rate=0.1, loss_function = 'MultiClass')
model.load_model('catboost_model3.dump')
@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page
    """
    with open("index.html", 'r') as viz_file:
        return viz_file.read()

@app.route("/score", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    data = flask.request.json
    color='#00dbfb'
    x = data["example"]
    score = model.predict_proba([x])
    print(score)
    list1=[]
    for i in score[0]:
        list1.append(i)
    index, value = max(enumerate(list1), key=operator.itemgetter(1))
    print(index)
    clas = ['F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+', 'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+']
    label = clas[index]
    if index <= 5:
        color ='#FF75A3'
    else:
        color ='#00dbfb'
    print(label)
    print(color)
    # Put the result in a nice dict so we can send it as json
    results = {"score":label,"color":color}
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
#app.run(host='0.0.0.0')
# app.run(debug=True)
