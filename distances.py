import pymongo
from itertools import combinations
from random import shuffle
from string_metrics import levenshtein, jaccard
from stats import distribution
from plotter import histogram
from IPython import embed # embed() to live shell


# DB SETTINGS
connection = pymongo.Connection()
db = connection['alternion']
dbprofiles = db.profiles.find({},{'_id':0})

# GENERATING PAIRS
# For each profile P
# There's a variable range of SNS, eg: P(twitter), P(facebook), P(tumblr)
# Creating all pair of SNSs [(tw,fb),(tw,tmblr),(fb,tmblr)]

profilesPairs = dict()
socialNetowrks = set()

def shuffleProfiles(profilePair):
  l1,l2 = zip(*profilePair)
  l = list(l2)
  shuffle(l)
  l2 = tuple(l)
  return  list(zip(l1,l2))


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



# COMPUTING ON PAIRS

class_counter = 0 # number of osn pair
pairs_counter = 0 # number of pairs
match_counter = 0 # number of exact match (levenshtein)
mean_counter = 0  # mean sum
jacmatch_counter = 0 # number of exact match (jaccard)
jacmean_counter = 0 # mean sum jaccard

all_distances = []
all_jac_distances = []

for sns,profilePairs in profilesPairs.items():
    nPairs = len(profilePairs)
    if nPairs > 1000 : # Plotting only if the class have more than 1000 pairs

      #shufflin all up
      profilePairs = shuffleProfiles(profilePairs)
      print("shuffled")

      #LEVENSHTEIN DISTANCES
      levdistances = [int(levenshtein(p)) for p in profilePairs] # cast to int() 'cause levenshtein return a numpy int
      distr = distribution(levdistances)
      distanceDistribution = [levdistances.count(x) for x in range(0,max(levdistances)+1)]
      exact_match = levdistances.count(0) # distance = 0

      #JACCARD DISTANCES
      jaccdistances = [jaccard(p) for p in profilePairs]
      jaccdistr = distribution(jaccdistances)
      jacexact_match = jaccdistances.count(0)

      #UPDATING OVERALL STATS
      all_distances.extend(levdistances)
      all_jac_distances.extend(jaccdistances)
      class_counter += 1
      mean_counter += distr[0]
      pairs_counter += len(profilePairs)
      match_counter += exact_match
      jacmatch_counter += jacexact_match
      jacmean_counter += jaccdistr[0]


      # #PLOTTING JACCARD CURRENT CLASS HISTOGRAM
      # title = str(sns)[1:-1].replace("'","").replace(",","-").replace(" ","")
      histogram(jaccdistances, 10, 'Distance', 'Username pairs', title, 'jac_plot/', range = ([0,1]))

      # #PLOTTING LEVENSTHEIN CURRENT  CLASS HISTOGRAM
      histogram(levdistances, max(levdistances), 'Distance', 'Username pairs', title, 'lev_plot/' )

      # #PRINT STATS
      # print("SNSs: {0} - #username pairs: {1}".format(sns,nPairs))
      # print("Distribution (Levenshtein distance): Mean, StandardDeviation, Median, Min, Max")
      # print(distr)
      # print("Exact match (Levenshtein distance = 0):")
      # print("#n: {0} - Percentage : {1}".format(exact_match, exact_match/nPairs))
      # print("Distance distribution:")
      # print(distanceDistribution)
      # print("Exact match (Jaccard distance = 0):")
      # print("#n: {0} - Percentage : {1}".format(jacexact_match, jacexact_match/nPairs))
      # print("\r\n")



#OVERALL  LEVENSHTEIN PLOT
histogram(all_distances,
          max(all_distances),
          'Distance',
          'Username pairs',
          'levenshtein distribution',
          'lev_plot/' )


#OVERALL JACCARD PLOT
histogram(all_jac_distances,
          10,
          'Distance',
          'Username pairs',
          'jaccard distribution',
          'jac_plot',
          range = ([0,1]))


#OVERALL STATS
print("Overall Mean LEVENSHTEIN - Sum(means) over {0} classes".format(class_counter))
print(mean_counter/class_counter)

print("Overall Mean JACCARD - Sum(means) over {0} classes".format(class_counter))
print(jacmean_counter/class_counter)

print("Overall Exact Match LEVENSHTEIN- Sum(exact_match) over {0} pairs".format(pairs_counter))
print(match_counter/pairs_counter)

print("Overall Exact Match JACCARD - Sum(exact_match) over {0} pairs".format(pairs_counter))
print(jacmatch_counter/pairs_counter)


print(socialNetowrks)
print(len(socialNetowrks))
