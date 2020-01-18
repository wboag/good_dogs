
from collections import defaultdict

from data import come_here_boy


def main():

    dogs = come_here_boy()

    counts = defaultdict(lambda:defaultdict(int))
    for dog in dogs:
        for key in ['Neighborhood', 'Dog_Name', 'Dog_Breed']:
            label = dog[key]
            counts[key][label] += 1

    for key in counts.keys():
        print key
        for label,count in sorted(counts[key].items(), key=lambda t: t[1])[-20:]:
            print '\t', count, label
        print


if __name__ == '__main__':
    main()
