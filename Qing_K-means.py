# CS 181, Spring 2016
# Homework 4: Clustering
# Name: Qing Zhao
# Email: qingzhao@g.harvard.edu

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

class KMeans(object):
	# K is the K in KMeans
	# useKMeansPP is a boolean. If True, you should initialize using KMeans++
	def __init__(self, K, useKMeansPP, D):
		self.K = K
		self.useKMeansPP = useKMeansPP
		self.D = D

	def cluster_points(self, X, mu):
		clusters  = {}
		for x in X:
			bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
						for i in enumerate(mu)], key=lambda t:t[1])[0]
			try:
				clusters[bestmukey].append(x)
			except KeyError:
				clusters[bestmukey] = [x]
		return clusters

	def reevaluate_centers(self, mu, clusters):
		newmu = []
		keys = sorted(clusters.keys())
		for k in keys:
			newmu.append(np.mean(clusters[k], axis = 0))
		return newmu

	def has_converged(self, mu, oldmu):
		return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

	def fit(self, X):
		# Initialize to K random centers
		self.X = np.array([pic.flatten() for pic in X])
		self.losses = []

		K = self.K
		oldmu = random.sample(self.X, K)

		if self.useKMeansPP == True:
			self.init_centers()
		else:
			self.mu = random.sample(self.X, K)

		while not self.has_converged(self.mu, oldmu):
			oldmu = self.mu
			# Assign all points in X to clusters
			self.clusters = self.cluster_points(self.X, self.mu)
			# Reevaluate centers
			self.mu = self.reevaluate_centers(oldmu, self.clusters)
			self.losses.append(self.compute_loss())

		fig = plt.plot(self.losses)
		if not self.useKMeansPP:
			plt.savefig('graphs/K{}Loss.png'.format(self.K,self.D))
		elif self.useKMeansPP:
			plt.savefig('graphs/K{}KPPLoss.png'.format(self.K,self.D))

	# This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
	def get_mean_images(self):
		return self.mu

	# This should return the arrays for D images from each cluster that are representative of the clusters.
	# How the representative images are selected?
	#
	def get_representative_images(self, D):
		cluster_repr = {}
		for i in self.clusters.keys():
			cluster_repr[i] = sorted(self.clusters[i], key=lambda x: np.linalg.norm(x - self.mu[i]))[:D]
		return cluster_repr

	# img_array should be a 2D (square) numpy array.
	# Note, you are welcome to change this function (including its arguments and return values) to suit your needs. 
	# However, we do ask that any images in your writeup be grayscale images, just as in this example.
	def create_image_from_array(self, img_array, title):
		plt.figure()
		img_array = img_array.reshape((28,28))
		plt.imshow(img_array, cmap='Greys_r')
		plt.title(title)
		plt.show()
		return

	def compute_loss(self):
		loss = 0
		for i in self.clusters.keys():
			for j in self.clusters[i]:
				loss += np.linalg.norm(j - self.mu[i])
		return loss

	def _dist_from_centers(self):
		cent = self.mu
		X = self.X
		D2 = np.array([min([np.linalg.norm(x-c)**2 for c in cent]) for x in X])
		self.D2 = D2

	def _choose_next_center(self):
		self.probs = self.D2/self.D2.sum()
		self.cumprobs = self.probs.cumsum()
		r = random.random()
		ind = np.where(self.cumprobs >= r)[0][0]
		return(self.X[ind])

	def init_centers(self):
		self.mu = random.sample(self.X, 1)
		while len(self.mu) < self.K:
			self._dist_from_centers()
			self.mu.append(self._choose_next_center())

	def show_reps(self,savefig = True):
		fig,ax = plt.subplots(self.K,self.D+1)
		fig.set_figwidth(20),fig.set_figheight(20)
		repr_images = KMeansClassifier.get_representative_images(self.D)

		for i in xrange(self.K):
			ax[i,0].imshow(self.mu[i].reshape((28,28)),cmap='Greys_r')

			repr = repr_images[i]
			for j in xrange(self.D):
				ax[i,j+1].imshow(repr[j].reshape((28,28)), cmap='Greys_r')
		if savefig:
			if not self.useKMeansPP:
				fig.savefig('graphs/RepImagesK{}D{}.png'.format(self.K,self.D), )
			elif self.useKMeansPP:
				fig.savefig('graphs/KPPRepImagesK{}D{}.png'.format(self.K,self.D),)
		plt.show()

	def show_mean(self, savefig = True):
		fig, ax = plt.subplots(1,self.K)
		fig.set_figwidth(20),fig.set_figheight(4)

		for i in xrange(self.K):
			img = self.mu[i].reshape((28,28))
			ax[i].imshow(img, cmap = 'Greys_r')
		if savefig:
			if not self.useKMeansPP:
				plt.savefig('graphs/meanK{}.png'.format(self.K))
			elif self.useKMeansPP:
				plt.savefig('graphs/KPPmeanK{}.png'.format(self.K))
		plt.show()

# This line loads the images for you. Don't change it! 
pics = np.load("images.npy")

# You are welcome to change anything below this line. This is just an example of how your code may look.
# That being said, keep in mind that you should not change the constructor for the KMeans class, 
# though you may add more public methods for things like the visualization if you want.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.

# TESTINGS

# k-maens,
k = 15
d = 5
KMeansClassifier = KMeans(K=k, useKMeansPP=False, D=d)
KMeansClassifier.fit(pics)
KMeansClassifier.show_mean()
KMeansClassifier.show_reps()



