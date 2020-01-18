
from collections import defaultdict
import numpy as np
import math

from data import come_here_boy


def main():

    #neighborhood_breed()
    #breed_clusters_one()
    breed_clusters_two()





def breed_clusters_two():

    dogs = come_here_boy()

    breeds = defaultdict(list)
    for dog in dogs:
        breed = dog['Dog_Breed']
        pt = map(float,dog['Location_masked'][1:-1].split(','))
        breeds[breed].append(pt)

    breeds = { k:np.array(v) for k,v in breeds.items() if len(v)>5 }
    #breeds = { k:np.array(v) for k,v in breeds.items() if len(v)>50 }
    #breeds = { k:np.array(v) for k,v in breeds.items() if 10<=len(v)<=11 }
    #breeds = { k:np.array(v) for k,v in breeds.items() if len(v)>100 }
    print len(breeds)
    #exit()

    # Look over all possible 0/1 masks to find a configuration that maximizes tightness & separation
    scores = {}
    #for candidate in brute_force(breeds.keys()):
    for candidate in breeds.keys():
        candidate = [candidate]
        '''
        if len(candidate) < 2:
            continue
        '''
        print candidate
        pts = { k:v for k,v in breeds.items() if k in candidate }
        scores[tuple(candidate)] = cluster_eval(pts)

    print
    for candidate,score in sorted(scores.items(), key=lambda t:t[1]):
        print score, candidate

    '''
    for breed,pts in sorted(breeds.items(), key=lambda t:cluster_eval(t[1])):
        print breed, cluster_measure(pts)
        print len(pts)
        print
    '''



'''
def brute_force(vals):
    if len(vals) == 0:
        return [[]]
    rest = brute_force(vals[1:])
    dont = rest
    do = [ [vals[0]]+r for r in rest ]
    return do + dont
'''

def brute_force(vals):
    n = len(vals)
    for i in range(2**n):
        inds = decompose(i)
        print i, inds
        candidate = [ vals[j] for j in inds ]
        #print candidate
        yield candidate


def decompose(n):
    i = n
    bits = []
    while i>0:
        if i%2:
            bits.append(1)
        else:
            bits.append(0)
        i /= 2
    #inds = [ 2**ind for ind,bit in enumerate(bits) if bit!=0 ]
    inds = [ ind for ind,bit in enumerate(bits) if bit!=0 ]
    return set(inds)



def cluster_eval(clusters):
    return centroid_penalty(clusters)
    #return dunn_index(clusters)



def centroid_penalty(clusters):
    centroids = { k:v.mean(axis=0) for k,v in clusters.items() }

    good_if_small = 0
    good_if_large = 0
    for label,pts in clusters.items():
        for pt in pts:
            good_if_small += euclidian(pt, centroids[label])

        '''
        for pt in pts:
            for other_label,C in centroids.items():
                if other_label == label: continue
                good_if_large += euclidian(pt, C)
        '''

    return good_if_large/len(clusters) - good_if_small


def dunn_index(clusters):
    closest_inter = float('inf')
    for c1,pts1 in clusters.items():
        for c2,pts2 in clusters.items():
            if c1 == c2: continue
            for pti in pts1:
                for ptj in pts2:
                    dist = euclidian(pti,ptj)
                    if dist < closest_inter:
                        closest_inter = dist

    farthest_intra = 0
    for c,pts in clusters.items():
        for pti in pts:
            for ptj in pts:
                dist = euclidian(pti,ptj)
                if dist > farthest_intra:
                    farthest_intra = dist

    return closest_inter / (farthest_intra + 1e-9)



def euclidian(u, v):
    diff = u-v
    return np.dot(diff.T, diff)**0.5


def breed_clusters_one():

    dogs = come_here_boy()

    breeds = defaultdict(list)
    for dog in dogs:
        breed = dog['Dog_Breed']
        pt = map(float,dog['Location_masked'][1:-1].split(','))
        breeds[breed].append(pt)

    breeds = { k:np.array(v) for k,v in breeds.items() if len(v)>3 }

    for breed,pts in sorted(breeds.items(), key=lambda t:cluster_measure(t[1])):
        print breed, cluster_measure(pts)
        print len(pts)
        print



def cluster_measure(pts):
    #pts /= pts.max(axis=0) - pts.min(axis=0)
    mu = pts.mean(axis=0)
    return (np.dot(mu.T, mu)) / len(pts)
    exit()
    pts = pts - mu

    dot = np.dot(pts.T, pts)
    val = (dot[0,0] + dot[1,1]) / len(pts)
    return -val



def neighborhood_breed():

    dogs = come_here_boy()

    counts = defaultdict(lambda:defaultdict(int))
    for dog in dogs:
        neighborhood = dog['Neighborhood']
        breed = dog['Dog_Breed']
        #counts[neighborhood][breed] += 1
        counts[breed][neighborhood] += 1

    counts = { k:v for k,v in counts.items() if sum(v.values())>5 }

    #for k1,hist in sorted(counts.items(), key=lambda t:entropy(t[1]), reverse=True):
    #for k1,hist in sorted(counts.items(), key=lambda t:len(t[1])):
    for k1,hist in sorted(counts.items(), key=lambda t:max(t[1].values())/float(sum(t[1].values()))):
        N = float(sum(hist.values()))
        print k1, int(N), entropy(hist)
        for k2,count in sorted(hist.items(), key=lambda t: t[1])[-20:]:
            print '\t', count/N,k2
        print


def entropy(hist):
    N = float(sum(hist.values()))
    dist = [ v/N for v in hist.values() ]
    ent = 0
    for p in dist:
        ent -= p * math.log(p)
    return ent


if __name__ == '__main__':
    main()
