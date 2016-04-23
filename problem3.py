# CS 181, Spring 2016
# Homework 4: Clustering
# Name:
# Email:
import time
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
class myKMeans(object):
        # K is the K in KMeans
        # useKMeansPP is a boolean. If True, you should initialize using KMeans++
    def __init__(self, K, useKMeansPP):
        self.K = K
        self.useKMeansPP = useKMeansPP
    # init a random pic to start
    def initMus(self,X):
        if not self.useKMeansPP:
            return X[np.random.randint(low = 0, high = self.num_pic, size = self.K)]
        elif self.useKMeansPP:
            return self.kpp(X)
        else:
            raise ValueError('Unknown')
    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def kpp(self, X):
        mus = np.zeros((self.K, X.shape[1], X.shape[2]))
        print "===============Using KMean++================\n"
        # loop over k
        
        for i in xrange(self.K):
        
        # k' < k == None, so we continue
            if i == 0:
                mus[i] = X[np.random.randint(low = 0, high = self.num_pic, size = 1)]
                continue
            
            # 6000 * k 
            ds = np.zeros((self.num_pic,i))

            for j in range(i):
                ds[:,j] = (np.linalg.norm(X - mus[j], axis=(1,2)))
 
            min_dist = np.min(ds,axis=1)
            print len(min_dist),min_dist
            normalization = np.sum(min_dist**2)
            p_s = min_dist**2/normalization
            mus[i] = X[np.random.choice(np.arange(0,self.num_pic),p=p_s)]
            
            #print mus
        return mus
#             print ds,len(ds),len(ds[0]),'\n'
    def fit(self, X):
        self.num_pic = X.shape[0]
        mus = self.initMus(X)
        # testing dimension of centers
        assert len(mus) == self.K
        assert mus[1].shape == X[1].shape
        m = 1000
        centeriods = mus
        #print mus
        j = []
        
        while True:
            # R is the assignment matrix
            # r_nk = {0,1}
            # r_n = one hot-encoded like [0,0,0,1,0,0......0]
            R = np.zeros((self.num_pic,self.K))
            
            assert np.sum(R) < 1
            
            # initialize the dist matrix 
            dist = np.zeros_like(R)
            
            for ks in xrange(self.K):
                dist[:,ks] = np.linalg.norm(X - mus[ks],axis=(1,2))
            for i in xrange(R.shape[0]):
                R[i,np.argmin(dist[i,:])] =1 

            
            for i in xrange(self.K):
                sel = np.array(R[:,i],dtype=np.bool)
                mus[i]= (np.mean(X[R[:,i]==1],axis = (0)))
            # keep track of error
            j.append(np.sum(dist[R.astype(np.bool)]))
            if len(j)>1:
                if j[-1] == j[-2]:
                    print 'Converged in {} iterations, find {} means\n-------- processing rep next ---------\n'.format(len(j),self.K)
                    break
            elif len(j)>1000:
                raise ValueError('maximum iteration reached')
            #print mus
        
        
        
        self.__dist__ = dist
        self.j = j
        self.mus = mus

        

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        assert len(self.mus) == self.K
        assert self.mus[0].shape == (28,28)
        return self.mus

    # This should return the arrays for D images from each cluster that are representative of the clusters.
    def get_representative_images(self, D):
        #print self.__dist__
        self.repidx = np.zeros((self.K,D))
        #print self.__dist__.shape
        for i in xrange(self.K):
            self.repidx[i] = np.argsort(self.__dist__[:,i])[:D]
        self.D = D
        return self.repidx

    # img_array should be a 2D (square) numpy array.
    # Note, you are welcome to change this function (including its arguments and return values) to suit your needs. 
    # However, we do ask that any images in your writeup be grayscale images, just as in this example.
    def create_image_from_array(self, img_array):
        plt.figure()
        plt.imshow(img_array, cmap='Greys_r')
        plt.show()
        return
    def show_trend(self,savefig = False):
        plt.figure()
        plt.plot(self.j)
        plt.title('Converged In {} iterations'.format(len(self.j)))
        if savefig:
            if self.useKMeansPP:
                plt.savefig('KPPTrendK{}'.format(self.K))
            else:
                plt.savefig('TrendK{}'.format(self.K))
    def show_reps(self,savefig = False):
        
        fig,ax = plt.subplots(self.K,self.D+1)
        fig.set_figwidth(20),fig.set_figheight(20)
        
        for i in xrange(self.K):
            ax[i,0].imshow(self.mus[i],cmap='Greys_r')
            for j in xrange(self.D):
                ax[i,j+1].imshow(pics[idxes[i][j]], cmap='Greys_r')
        if savefig:
            if not self.useKMeansPP:
                fig.savefig('RepImagesK{}D{}.png'.format(self.K,self.D), )
            elif self.useKMeansPP:
                fig.savefig('KPPRepImagesK{}D{}.png'.format(self.K,self.D),)
        plt.show()
        


    def show_mean(self, savefig = False):
        fig, ax = plt.subplots(1,self.K)
        fig.set_figwidth(20),fig.set_figheight(4)
        
        for i in xrange(self.K):
            ax[i].imshow(self.mus[i], cmap = 'Greys_r')
        if savefig:
            if not self.useKMeansPP:
                plt.savefig('meanK{}.png'.format(self.K))
            elif self.useKMeansPP:
                plt.savefig('KPPmeanK().png'.format(self.K))
        plt.show()

# This line loads the images for you. Don't change it! 
pics = np.load("images.npy", allow_pickle=False)
'''
# You are welcome to change anything below this line. This is just an example of how your code may look.
# That being said, keep in mind that you should not change the constructor for the KMeans class, 
# though you may add more public methods for things like the visualization if you want.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.
K = 15
KMeansClassifier = KMeans(K, useKMeansPP=False)
KMeansClassifier.fit(pics)# for i in KMeansClassifier.mus:
KMeansClassifier.show_trend(False)
KMeansClassifier.show_mean(savefig = False)
idxes = KMeansClassifier.get_representative_images(8)

KMeansClassifier.show_reps(False)
'''
sizes = np.arange(0, 6000, 300)
clusters = np.arange(1, 15, 1)
times = np.empty(len(clusters))
times2 = np.empty(len(clusters))

def wrapper_scikit(K):
    pics_t = np.empty((pics.shape[0],np.power(pics.shape[1],2)))
    for i in range(pics_t.shape[0]):
        pics_t[i] = pics[i].flatten()
    time1 = time.time()
    kmean = KMeans(init='random', n_clusters=K)
    kmean.fit_transform(pics_t)
    time2 = time.time()
    return (time2-time1)*1000.

def wrapper_kmean(K):
    time1 = time.time()
    KMeansClassifier = myKMeans(K, useKMeansPP=False)
    KMeansClassifier.fit(pics)
    time2 = time.time()
    return (time2-time1)*1000.

for i in range(1, len(clusters)):
    times[i] = wrapper_kmean(clusters[i])
    times2[i] = wrapper_scikit(clusters[i])
    print times[i], times2[i]

plt.xlabel('Number of clusters')
plt.ylabel('Time in milliseconds')
plt.plot(clusters, times,label=r'Python Implementation')
plt.plot(clusters, times2, label=r'Scikit-learn Implementation')
plt.legend(loc='best')
plt.show()


