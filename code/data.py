
import os
from os.path import dirname
import re


def come_here_boy():
    return load_csv('Dogs_of_Cambridge.csv')


def crimes():
    return load_csv('Crime_Reports.csv')


def load_csv(filename):
    dogs = []
    base_dir = dirname(dirname(os.path.abspath(__file__)))
    dogs_file = os.path.join(base_dir, 'data', filename)
    with open(dogs_file, 'r') as f:
        keys = f.readline().strip().split(',')
        text = f.read()
        matches = re.findall('(.*?),(.*?),"(\(.*?,.*?\))",(.*?),(.*?),(.*)', text)
        for match in matches:
            assert len(keys) == len(match)
            dog = dict(zip(keys,match))
            dogs.append(dog)
    return dogs


def theyre_good_dogs_brent():
    base_dir = dirname(dirname(os.path.abspath(__file__)))
    ratings_file = os.path.join(base_dir, 'data', 'we_rate_dogs.tsv')
    with open(ratings_file, 'r') as f:
        ratings = []
        keys = f.readline().strip('\n\r').split('\t')
        for line in f.readlines():
            line = clean_text(line)
            toks = line.strip('\n\r').split('\t')
            assert len(keys) == len(toks)
            dog = dict(zip(keys,toks))
            ratings.append(dog)
    return ratings


def clean_text(text):
    try:
        return text.encode('ascii', 'ignore')
    except Exception as e:
        ret = []
        for c in text:
            try:
                ret.append(c.encode('ascii', 'ignore'))
            except Exception as f:
                pass
        return ''.join(ret)
