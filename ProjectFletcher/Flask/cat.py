
import pandas as pd
import gensim
import os
import collections
import smart_open
import flask
from flask import Flask
import pickle
import random
from surprise import SVDpp,SVD
from surprise import Dataset
from surprise import Reader
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import euclidean_distances
​
app = flask.Flask(__name__)
​
#pulling in dataset, item list
with open('/Users/deven/Documents/pickleddata/projectfletcher/newdatalist.pkl', 'rb') as picklefile:
    itemdf = pickle.load(picklefile)
with open('/Users/deven/Documents/GitHub/Metis/ProjectFletcher/surprise_data.pkl', 'rb') as picklefile:
    newdf = pickle.load(picklefile)
with open('/Users/deven/Documents/pickleddata/projectfletcher/btrain.pkl', 'rb') as picklefile:
    btrain = pickle.load(picklefile)
with open('doclen.pkl', 'rb') as picklefile:
    doclen = pickle.load(picklefile)
#changing some tea names for easier calls
itemdf = pd.DataFrame(itemdf)
newname=[]
import re
​
for i in itemdf['Tea Name']:
    line = re.sub('[!@#$\'\",]', '', i)
    newname.append(line)
itemdf['Tea Name'] = newname
​
#initializing book list
bookt = ['Emma by Jane Austen', 'Persuassion by Jane Austen', 'Sense and Sensibility by Jane Austen',\
        'Poems by William Blake', 'The Little People of the Snow by William Bryant', 'The Adventures of Buster Bear by Thornton Burgress'\
        'Alice in Wonderland by Lewis Carroll','The Ball and the Cross by G.K. Chesterton','The Wisdom of Father Brown by G.K. Chesterton'\
        'The Ball and the Cross by G.K. Chesterton', 'The Parents Assistant by Maria Edgeworth','Moby Dick by Herman Melville',\
        'Paradise Lost by John Milton', 'Shakespeares Works','Shakespeares Works','Shakespeares Works', 'Leaves of Grass by Walt Whitman']
​
imagelocs = {'Emma by Jane Austen':'static/emma.jpg','Persuassion by Jane Austen':'static/persuassion.jpg','Sense and Sensibility by Jane Austen':'static/sense.jpg',\
'Poems by William Blake':'static/blake.jpg','The Little People of the Snow by William Bryant':'static/bryant.jpg','The Adventures of Buster Bear by Thornton Burgress':'static/the-adventures-of-buster-bear.jpg',\
'Alice in Wonderland by Lewis Carroll':'static/alice.jpg', 'The Ball and the Cross by G.K. Chesterton':'static/ball.jpg','The Wisdom of Father Brown by G.K. Chesterton':'static/fatherbrown.jpg',\
'The Parents Assistant by Maria Edgeworth':'static/edge.jpg','Moby Dick by Herman Melville':'static/moby.jpg','Paradise Lost by John Milton':'static/paradise.jpg',\
'Shakespeares Works':'static/shake.jpg','Shakespeares Works':'static/shake.jpg','Shakespeares Works':'static/shake.jpg','Leaves of Grass by Walt Whitman':'static/leaves.jpg'}
​
​
#Call function for top ratings
from collections import defaultdict
def get_top_n(teaid,score,qq, n=3):
    # A reader is still needed but only the rating_scale param is requiered.
    qq = pd.concat([qq,pd.DataFrame([[score,teaid, 'user1']], columns = ['Score', 'Tea Name', 'User Name'])], ignore_index=True)
    reader = Reader(rating_scale=(0, 100))
    algo=SVD()
    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(newdf[['User Name', 'Tea Name', 'Score']], reader)
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)


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
    want= []
    for i in predictions:
        if i[0]== 'user1':
            want.append(i)
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in want:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    teadist = []
    mindist = []
    eudist=0
    for i in top_n['user1']:
        eudis=(euclidean_distances(teadf.iloc[itemdf[itemdf['Tea Name']==x[0]].index,:], \
            teadf.iloc[itemdf[itemdf['Tea Name']==i[0]].index,:]))
        teadist.append((i[0],eudist))
    mindist = sorted(teadist, key=lambda x:x[1])

    return mindist



def getBookrec(iid):
    bookrec = gensim.models.doc2vec.Doc2Vec.load('/Users/deven/Documents/pickleddata/projectfletcher/bookrec.bin')
    test_corpus = itemdf[itemdf['Tea Name']==iid]['Review Adj'].values[0]
    inferred_vector = bookrec.infer_vector(test_corpus)
    sims = bookrec.docvecs.most_similar([inferred_vector])
    rec=''
    tot=0
    for ind, i in enumerate(doclen):
        tot+=i
        if sims[0][0]<tot:
            rec = bookt[ind-1]
            break
    return rec
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
    x = data["example"]
    rec1,rec2,rec3 = get_top_n(x[0],x[2], newdf)
    bookrec= getBookrec(x[0])
    imgrec = imagelocs[bookrec]
    # Put the result in a nice dict so we can send it as json
    results = {"tearec1":rec1,"tearec2":rec2,"tearec3":rec3,"bookrec":bookrec, 'img':img}
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
#app.run(host='0.0.0.0')
# app.run(debug=True)
