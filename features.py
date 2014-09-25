from  __future__ import division
from functools import reduce
import math
import string
from stats import distribution
from keyboard import *

alphabet = string.ascii_lowercase

### METRICS FUNCTIONS ###
### PATTERNS DUE TO HUMAN LIMATIONS ###

# Username lenght likelihood
def ull(usernames):
  return distribution(list(len(x) for x in usernames))

# Unique username creation likelihood
def uucl(usernames):
  return len(set(usernames)) / len(usernames)


### EXOGENOUS FACTORS ###
### TYPING PATTERNS ###

def biGrams(word):
  return [[word[x],word[x+1]] for x in range(0, len(word)-1)]

# keys is a pair of key e.g ('a','q')
# Return boolean or the string rapresenting the hand used
def sameHand(keys, handInfo = False):
  lefthand,righthand = left_hand(), right_hand()
  if not handInfo:
    return ( keys[0] in lefthand and keys[1] in lefthand ) or ( keys[0] in righthand and keys[1] in righthand )
  else:
    return ( keys[0] in lefthand and keys[1] in lefthand and 'left') or ( keys[0] in righthand and keys[1] in righthand and 'right')

def sameFinger(keys):
  if sameHand(keys):
    samefinger = [all((keys[0] in finger, keys[1] in finger)) for finger in typing_map[sameHand(keys,True)].values()]
    return sum(samefinger) > 0 and True or False
  else:
    return False

# The percentage of keys typed using the same (X) used for the previous key.
# (X) depending on the granularities e.g 'Hand' or 'Finger'
def sameRate(username, granularitiesFunction):
  username = username.replace(" ","").lower()
  bigram = biGrams(username)
  samerate = [granularitiesFunction(bg) for bg in bigram]
  return sum(samerate) / (len(username) -1)

# The percentage of keys typed using each finger order by hands order by finger (left-right/index,middle,pinkie,ring)
def eachFingerRate(username):
  to_flat = [[(finger, hand, sum([username.count(key)
            for key in typing_map[hand][finger]])/len(username))
            for finger in typing_map[hand]]
            for hand in typing_map.keys()]
  ordered = sorted([rate for hand in to_flat for rate in hand], key = lambda tup: (tup[0],tup[1]))
  return [el[2] for el in ordered]

#The percentage of keys pressed on rows: Top Row, Home Row, Bottom Row, and Number Row
def rowsRate(username):
  return [sum([c in row for c in username]) for row in typing_row]

# The approximate distance (in meters) traveled for typing a username
# Normal typing keys are assumed to be (1.8cm)^2 (including gap between keys).
def travelledDistance(username):
  pass

### ENDOGENOUS FACTORS ###

def alphabetDistribution(username):
  return [username.count(c)/len(username) for c in alphabet]

def shannonEntropy(text):
  distribution = alphabetDistribution(text)
  return reduce((lambda x,y: x - (y * math.log(y,2) if y > 0 else 0)), distribution, 0)

def naivEntropy(text):
  text = set(text).intersection(set(alphabet))
  return len(text) / len(alphabet)


# Longest Common Substring - data is a collection of strings, eg : ['mattia','mattiadmr']
# If normalized return lcs lenght values in range [0,1] (normalized by the maximum length of the two\n strings)
# Usefull to catch prefixes - suffixes
def lcsubstring(data, normalized = False):
  substr = ''
  if len(data) > 1 and len(data[0]) > 0:
    for i in range(len(data[0])):
      for j in range(len(data[0])-i+1):
        if j > len(substr) and all(data[0][i:i+j] in x for x in data):
          substr = data[0][i:i+j]
  if normalized:
    return len(substr) / max([len(d) for d in data])
  return substr

# Longest Common Subsequence
# Usefull to detect abbreviations
def lcs(a, b, normalized = False):
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = \
                    max(lengths[i+1][j], lengths[i][j+1])
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            assert a[x-1] == b[y-1]
            result = a[x-1] + result
            x -= 1
            y -= 1
    if normalized:
      return len(result) / max([len(a),len(b)])
    return result

# Dynamic Time Warping
# TODO : How to apply this to strings? Alignment on time makes any sense on strings?
def dtw(data):
  pass


