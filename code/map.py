
import numpy as np
import pylab as plt

from data import come_here_boy


def main():

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

    '''
    breeds = {
             #'Labradoddle':'blue',
             #'Puggle':'red',
             #'Jack Russel Terrier':'green',
             #-------------------------------
             'Scottish Terrier':'blue',
             'German Shepherd Mix':'red',
             #-------------------------------
             #'Yorkie Terrier':'red',
             #'Italian Greyhould':'blue',
             #'Australian Terrier':'green',
             ##'Jindo':'purple',
             ##'Lakeland Terrier':'fuchsia',
             ##'Weimaramer':'orange',
             #-------------------------------
             #'Golden Retriever':'red',
             #'Chihuahua':'green',
             #-------------------------------
             #'Labrador Retriever':'blue',
             #'Golden Retriever':'red',
             #'Chihuahua':'green',
             #'Mix':'purple',
             #'Terrier':'fuchsia',
             #'Shih Tzu':'orange',
             #'Poodle':'black',
             #'Beagle':'yellow',
             #'German Shepherd':'lime',
             #'Dachshund':'chocolate',
             }
    '''
    #breeds_labels = ('German Shepherd Mix', 'Miniature Dachshund')
    breeds_labels = ('Yorkie Terrier', 'Airedale', 'Pitbull')
    colors = ['blue','green','red','purple','fuchsia','orange','yellow','black','cyan','chocolate']
    breeds = dict(zip(breeds_labels,colors))

    names = {
             'Bella':'blue',
             'Lucy':'red',
             'Charlie':'green',
             'Daisy':'purple',
             'Max':'yellow',
             'Lola':'orange'
             }

    target = 'Neighborhood'
    targets = neighborhoods
    #target = 'Dog_Name'
    #targets = names
    #target = 'Dog_Breed'
    #targets = breeds

    X,Y,colors = [],[],[]
    for dog in dogs:
        y,x = dog['Location_masked'][1:-1].split(',')

        if dog[target] in targets:
            X.append(float(x))
            Y.append(-float(y))
            colors.append( targets[dog[target]] )

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

    #ax.scatter(X, Y, c=colors)
    ax.scatter(X, Y, c=colors, alpha=0.2)
    plt.show()




if __name__ == '__main__':
    main()
