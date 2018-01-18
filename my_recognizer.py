import warnings
from asl_data import SinglesData
import numpy as np


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for item in range(test_set.num_items):

      word_X, word_lengths = test_set.get_item_Xlengths(item)
      scores = {}

      best_word = None
      best_score = -np.Inf

      for word in models:
        model = models[word]
        try:
          score = model.score(word_X, word_lengths)
          scores[word] = score

          if score>best_score:
            best_score = score
            best_word = word

        except:
          scores[word] = -np.Inf

      probabilities.append(scores)
      guesses.append(best_word)

    return probabilities, guesses
