import pandas as pd

from surprise import NormalPredictor
from surprise import SVDpp,SVD
from surprise import Dataset
from surprise import Reader
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#import urllib2

app = flask.Flask(__name__)

#pulling in dataset, item list
with open('newdatalist.pkl', 'rb') as picklefile:
    itemdf = pickle.load(picklefile)
with open('surprise_data.pkl', 'rb') as picklefile:
    surprisedf = pickle.load(picklefile)

#changing some tea names for easier calls
itemdf = pd.DataFrame(tea_list)
newname=[]
import re

for i in itemdf['Tea Name']:
    line = re.sub('[!@#$\'\",]', '', i)
    newname.append(line)
itemdf['Tea Name'] = newname



#Call function for top ratings
from collections import defaultdict
def get_top_n(teaid,score, n=10):
    #Surprise SVD
    # A reader is still needed but only the rating_scale param is requiered.
    newdf = pd.concat([newdf,pd.DataFrame([[score,teaid, 'user1']], columns = ['Score', 'Tea Name', 'User Name'])], ignore_index=True)
    reader = Reader(rating_scale=(0, 100))
    algo=SVD()
    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(newdf[['User Name', 'Tea Name', 'Score']], reader)
    trainset = data.build_full_trainset()
    algo.fit(trainset)


    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

        eudis=(euclidean_distances(newteaprofiledf[itemdf['Tea Name']==i]['Flavor Profile Reviews'], \
        newteaprofiledf[newteaprofiledf['Tea Name']==k[0]]['Flavor Profile Reviews']))
    return top_n


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
