
import numpy as np
import pylab as plt
import sys
from collections import defaultdict

from data import come_here_boy
from generate_ratings import build_generator


def main():

    rater = build_generator()

    dogs = come_here_boy()

    neighborhoods = {
                     'West Cambridge':'blue',
                     'North Cambridge':'red',
                     'Neighborhood Nine':'green',
                     'Cambridgeport':'purple',
                     'East Cambridge':'fuchsia',
                     'Mid-Cambridge':'orange',
                     'Wellington-Harrington':'black',
                     'The Port':'yellow',
                     'Riverside':'lime',
                     'Agassiz':'chocolate',
                     'Cambridge Highlands':'lightsalmon',
                     'Strawberry Hill':'deepskyblue',
                     'NA':'goldenrod',
                     'Area 2/MIT':'seagreen',
                     }

    target = 'Neighborhood'
    targets = neighborhoods

    X,Y,colors,ratings = [],[],[],[]
    neighborhood_ratings = defaultdict(list)
    for i,dog in enumerate(dogs):
        #if i%5 in [1,2,3,4]: continue

        y,x = dog['Location_masked'][1:-1].split(',')

        if dog[target] in targets:
            X.append(float(x))
            Y.append(-float(y))
            colors.append( targets[dog[target]] )

            rating = rater.rate(dog['Dog_Name'])
            ratings.append(rating)

            neighborhood_ratings[dog['Neighborhood']].append(rating)

            #print rating, dog['Dog_Name']


    # population statistics
    print
    rv = np.array(ratings)
    pop_mu    = rv.mean()
    pop_sigma = rv.std()
    print 'mu:', pop_mu
    print 'sigma:', pop_sigma
    pop_N = len(ratings)
    lo = pop_mu - 2*pop_sigma - .02
    hi = pop_mu + 2*pop_sigma + .02
    pop_2sigma = len([r for r in ratings if (lo <= r <= hi)])
    print '[%.3f,%.3f]  %d/%d (%.4f)' % (lo,hi,pop_2sigma,pop_N,float(pop_2sigma)/pop_N)
    print

    if '--hist' in sys.argv:
        plt.hist(ratings, bins=100)
        plt.title('Frequencies of Cambridge Dog Ratings')
        plt.ylabel('Number of Dogs')
        plt.xlabel('Rating')
        plt.axis([11.4,14.2,0,500])
        plt.show()
        exit()


    print
    for neighborhood,rs in sorted(neighborhood_ratings.items(), key=lambda t:np.array(t[1]).mean()):
        R = np.array(rs)
        N = len(rs)
        print '%-22s(%3d)  %.3f %.3f %.3f %.5f' % (neighborhood, N, R.min(), R.max(), R.mean(), R.std())
    print

    # intensity based on colors
    ratings_np = np.array(ratings)
    mu_rating = ratings_np.mean()
    diff_ratings = ratings_np - mu_rating
    diff_max = diff_ratings.max()
    diff_min = diff_ratings.min()
    R_ = diff_ratings / (diff_max-diff_min)
    R = R_ - R_.min()
    Z = R**1 / R.std()
    intensity = Z / (Z.max()-Z.min())
    #intensity = Z
    print intensity
    print intensity.min(), intensity.max()
    print intensity.mean(), intensity.std()
    #exit()

    #colors = intensity*255


    X = np.array(X)
    Y = np.array(Y)

    #mu_x = X.mean()
    #mu_y = Y.mean()
    mu_x = -71.118086221
    mu_y = -42.3774328989

    #diff_x = X.max() - X.min()
    #diff_y = Y.max() - Y.min()
    diff_x = 0.11382209
    diff_y = 0.05093936

    # center the data
    X -= mu_x
    Y -= mu_y

    # scale to 0-1
    X /= diff_x
    Y /= diff_y

    mu_x = X.mean()
    mu_y = Y.mean()

    img = plt.imread('../data/map_cambridge.png')
    fig, ax = plt.subplots()
    ax.imshow(img)

    xsize,ysize,_ = img.shape

    X *= 1200
    Y *= 800

    X += 590
    Y += 500

    bottom_20_percent = sorted(colors)[int(len(colors)*.10)]
    top_20_percent    = sorted(colors)[int(len(colors)*.90)]

    '''
    #ids = [i for i,val in enumerate(intensity) if val>0.5]
    #ids = [i for i,val in enumerate(colors) if val<50]
    ids = [i for i,val in enumerate(colors) if val<=bottom_20_percent]
    #ids = [i for i,val in enumerate(colors) if val>=top_20_percent]

    X = X[ids]
    Y = Y[ids]
    colors = colors[ids]
    '''

    print colors
    print len(colors)

    #ax.scatter(X, Y, c=colors, alpha=0.1)
    ax.scatter(X, Y, c=colors, alpha=1)
    plt.show()




if __name__ == '__main__':
    main()
