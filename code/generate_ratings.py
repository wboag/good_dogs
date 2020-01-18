
import numpy as np
import pylab as plt
import nltk
import re
import sys
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Ridge

from data import theyre_good_dogs_brent


def main():

    generator = build_generator()
    names = ['Willie', 'Peter', 'Andrew', 'Neil', 'Dom', 'Maggie', 'Alyssa', 'Beth', 'Katie']
    for name in names:
        print generator.rate(name), name
    print '\n'
    print generator.rate(sys.argv[1])



def build_generator():

    # load the data
    dogs = theyre_good_dogs_brent()

    # inputs and outputs
    names = []
    scores = []
    for dog in dogs:
        names.append(dog['Dog'])
        scores.append(dog['Rating'])

    # fit model
    generator = RatingGenerator()
    generator.fit(names, scores)

    return generator



class RatingGenerator:
    def __init__(self):
        pass

    def fit(self, names, ratings):
        # Extract features from names for prediction
        features = map(extract_name_features,names)

        # Extract numeric ratings
        Y = np.array([int(re.search('(\d+)/10', rating).groups()[0]) for rating in ratings])

        # Vectorize the data
        self._vect = DictVectorizer()
        X = self._vect.fit_transform(features)

        # Fit the model
        #self._lr = LinearRegression(C=1.0)
        self._lr = Ridge(alpha=1e-4, normalize=True)
        self._lr.fit(X, Y)
        #print self._lr.coef_
        #print self._lr.intercept_

    def rate(self, name):
        features = extract_name_features(name)
        X = self._vect.transform(features)
        #return int(round(self._lr.predict(X)[0]))
        return self._lr.predict(X)[0]



def extract_name_features(name):
    features = defaultdict(int)
    features['len'] = len(name)
    name = name.lower()
    for n in [1,2,3,4]:
        for i in range(len(name)-n+1):
            ngram = name[i:i+n]
            featname = ('%d-gram'%n, ngram)
            features[featname] += 1
    return dict(features)



class TextGenerator:
    def __init__(self, names, texts):
        self._intros = defaultdict(int)
        self._ratings = defaultdict(int)
        self._middles = defaultdict(list)
        self._ends = defaultdict(list)

        for name,text in zip(names,texts):
            print name
            for sent in nltk.sent_tokenize(text):
                print '\t', sent
            print

            sents = nltk.sent_tokenize(text)

            # intro
            intro = sents[0].replace(name, '__name__')
            self._intros[intro] += 1

            # middle
            index = find_rating(sents)
            middle = '\n'.join(sents[1:index])

            # generate rating
            score = int(re.search('(\d+)/10', text).groups()[0])
            self._ratings[score] += 1

            # relevant description?
            end = '\n'.join(sents[index:]).replace('%d/10'%score, '__score__')
            print end

        print '\n\n'



def find_rating(lines):
    for index,line in enumerate(lines):
        if re.search('(\d+)/10', line):
            return index


def build_ngrams(text):
    print text
    exit()


if __name__ == '__main__':
    main()
