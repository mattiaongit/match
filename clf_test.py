import sklearn
from features import *
from sklearn import svm, cross_validation
import sklearn.metrics
from sklearn import preprocessing
from string_metrics import *
from random import shuffle


import pymongo
from itertools import combinations



#IMPORT DATA FROM DB
connection = pymongo.Connection()
db = connection['alternion']
dbprofiles = db.profiles.find({},{'_id':0})

profilesPairs = dict()
socialNetowrks = set()


# UTILS
def shuffleProfiles(profilePair):
  l1,l2 = zip(*profilePair)
  l = list(l2)
  shuffle(l)
  l2 = tuple(l)
  return  list(zip(l1,l2))

# TAKE A USERNAME PAIR AND RETURN A FEATURES VECTOR
def vectorize(pair):
  pairdata = []
  pairdata.extend(ull(pair))
  pairdata.extend(alphabetDistribution(pair[0]))
  pairdata.extend(alphabetDistribution(pair[1]))
  pairdata.extend(eachFingerRate(pair[0]))
  pairdata.extend(eachFingerRate(pair[1]))
  pairdata.extend(rowsRate(pair[0]))
  pairdata.extend(rowsRate(pair[1]))
  pairdata.append(sameRate(pair[0], sameFinger))
  pairdata.append(sameRate(pair[1], sameFinger))
  pairdata.append(sameRate(pair[0], sameHand))
  pairdata.append(sameRate(pair[1], sameHand))
  pairdata.append(jaccard(pair))
  pairdata.append(levenshtein(pair))
  pairdata.append(shannonEntropy(pair[0]))
  pairdata.append(shannonEntropy(pair[1]))
  pairdata.append(lcsubstring(pair,True))
  pairdata.append(lcs(pair[0],pair[1],True))
  return pairdata


#POPULATE DICT WITH USERNAME PAIR FOR EACH SOCIAL NETWORK
for profile in list(dbprofiles):
  profilePairs = list(combinations(profile.items(),2))
  for pair in profilePairs:
    sn1 = pair[0][0].capitalize()
    sn2 = pair[1][0].capitalize()
    socialNetowrks.update(set((sn1,sn2)))
    key = tuple(sorted((sn1,sn2)))
    if key not in profilesPairs.keys():
        profilesPairs[key] = []
    profilesPairs[key].append((pair[0][1]['username'],pair[1][1]['username']))


# Extract pairs of a specific social networks pair
dataset = profilesPairs[('Facebook','Google+')]
dataset = [pair for pair in dataset if len(pair[0]) > 0 and len(pair[1]) > 0]


raw_data = dataset + shuffleProfiles(dataset)
# LABELS OF DATASET (1: positive match, 0: negative match)
# THE SHUFFLED USERNAME PAIRS IS GOING TO BE LABELLED AS 0
targets = [1] * len(dataset) + [0] * len(dataset)
data = []

# BUILDING FEATURES INPUT VECTOR
for sample in raw_data:
  data.append(vectorize(sample))

# CROSS VALIDATION
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
  data, targets, test_size = len(data)/2 , random_state = 0 )

# DATA NORMALIZATION
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

#TRAIN
clf = svm.SVC(gamma = 0.001, C=50)
clf.fit(X_train,y_train)
y_pred = []

#TEST
for t in X_test:
  y_pred.append(clf.predict(t))

# OUTPUT CLASSIFICATOR SCORES
precision = sklearn.metrics.precision_score(y_test,y_pred)
recall = sklearn.metrics.recall_score(y_test,y_pred)
f1 = 2 * ( (precision * recall) / (precision + recall) )
print("{0} prec: {1}, rec: {2}, f1: {3}".format(key,precision,recall,f1))

