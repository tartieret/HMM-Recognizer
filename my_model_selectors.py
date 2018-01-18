import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = np.Inf
        best_model = None

        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)

                state_params = model.transmat_.shape[0] * (model.transmat_.shape[1] - 1)
                output_params = n_components * self.X.shape[1] * 2 # As it is a Gaussian HMM
                total_params = state_params + output_params + (n - 1)

                # Use parameters = n*n+2*n*d
                parameters = n_components*(n_components-1)+2*self.X.shape[1]*n_components
                score = -2*logL+total_params*np.log(self.X.shape[0])

                if score<best_score:
                    best_score = score
                    best_model = model

            except:
                pass

        if best_model is None:
            return self.base_model(self.min_n_components)
        else:
            return best_model



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = -np.Inf
        best_model = None

        word_list = list(self.words)
        word_list.remove(self.this_word)
        nb_words = len(self.words)

        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                total = 0

                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)

                for word in word_list:
                    X, lengths = self.hwords[word]
                    total += model.score(X,lengths)

                score = logL-total/(nb_words-1)

                if best_score<score:
                    best_score = score
                    best_model = model
            except:
                pass

        if best_model is None:
            return self.base_model(self.min_n_components)
        else:
            return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        if len(self.sequences) == 1:
            return self.base_model(self.n_constant)

        best_score = -np.Inf
        best_model = None

        # iterate through the range of components
        for n_components in range(self.min_n_components, self.max_n_components+1):
            scores = []

            # use 3-folds cross validation
            n_splits = min(3,len(self.sequences))
            kfold = KFold(random_state=self.random_state, n_splits = n_splits)
            for train_idx, test_idx in kfold.split(self.sequences):

                # split the data in training and cross validation set
                x_train, lengths_train = combine_sequences(train_idx, self.sequences)
                x_test, lengths_test = combine_sequences(test_idx, self.sequences)

                # train the model
                try:
                    model = GaussianHMM(n_components=n_components, n_iter=1000).fit(x_train, lengths_train)

                    # model scoring
                    logL = model.score(x_test, lengths_test)
                    scores.append(logL)
                except:
                    pass

            mean_score = 0
            if len(scores)>0:
                mean_score = np.average(scores)
            
                # if the average score is better than the
                # best score, then keep the model
                if mean_score>best_score:
                    best_score = mean_score
                    best_model = model

        if best_model is None:
            return self.base_model(self.n_constant)
        else:
            return best_model

