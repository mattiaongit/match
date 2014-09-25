import matplotlib as mpl
mpl.use('Agg') # Avoid no screen problem on server, no -X option needed
import matplotlib.pyplot as plt

extension = ".png"

def histogram(data, bins, xlabel, ylabel, title, savepath, range = None):
  plt.hist(data, bins = bins, range = range )
  plt.xlabel(xlabel, fontsize=20)
  plt.ylabel(ylabel, fontsize= 20)
  plt.title(title)
  plt.savefig(savepath+title+extension)
  plt.clf()
