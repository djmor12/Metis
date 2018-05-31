import pandas as pd
import gensim
import os
import collections
import smart_open
import random
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
with open('/Users/deven/Documents/pickleddata/projectfletcher/newdatalist.pkl', 'rb') as picklefile:
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

#initializing book list
bookt = ['Emma by Jane Austen', 'Persuassion by Jane Austen', 'Sense and Sensibility by Jane Austen',\
        'Poems by William Blake', 'The Little People of the Snow by William Bryant', 'The Adventures of Buster Bear by Thornton Burgress'\
        'Alice in Wonderland by Lewis Carroll','The Ball and the Cross by G.K. Chesterton','The Wisdom of Father Brown by G.K. Chesterton'\
        'The Ball and the Cross by G.K. Chesterton', 'The Parents Assistant by Maria Edgeworth','Moby Dick by Herman Melville',\
        'Paradise Lost by John Milton', 'Shakespeares Works','Shakespeares Works','Shakespeares Works', 'Leaves of Grass by Walt Whitman']

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
    teadist = []
    mindist = []
    for i in top_n['user1']:
        eudis=(euclidean_distances(itemdf[itemdf['Tea Name']==id]['Flavor Profile Reviews'], \
        itemdf[itemdf['Tea Name']==i[0]]['Flavor Profile Reviews']))
        teadist.append((i[0],eudist))
    mindist = sorted(teadist, key=lambda x:x[1])

    return mindist[0],mindist[1],mindist[2]

def getBookrec(iid):
    test_corpus = itemdf[itemdf['Tea Name']==iid]['Review Adj'].values
    bookrec = gensim.models.doc2vec.Doc2Vec.load('bookrec.bin')
    inferred_vector = bookrec.infer_vector(test_corpus)
    sims = bookrec.docvecs.most_similar([inferred_vector])

    tot=0
    for ind, i in enumerate(doclen):
        tot+=i
        if sims[0][0]==btrain[ind][1]:
            rec = bookt[ind-1]
            break
    return rec

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
    rec1,rec2,rec3 = get_top_n(x[0],x[2])
    bookrec= getBookrec(x[0])
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
    results = {"tearec1":rec1,"tearec2":rec2,"tearec3":rec3,"bookrec":bookrec}
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
#app.run(host='0.0.0.0')
# app.run(debug=True)
