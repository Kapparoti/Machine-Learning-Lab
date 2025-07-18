import numpy as np
from datasets import two_moon_dataset, gaussians_dataset
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.ion()


def spectral_clustering(data, n_cl, sigma=1., fiedler_solution=False):
	"""
	Spectral clustering.

	Parameters
	----------
	data: ndarray
		data to partition, has shape (n_samples, dimensionality).
	n_cl: int
		number of clusters.
	sigma: float
		std of radial basis function kernel.
	fiedler_solution: bool
		return fiedler solution instead of kmeans

	Returns
	-------
	ndarray
		computed assignment. Has shape (n_samples,)
	"""
	if fiedler_solution and n_cl != 2:
		raise Exception("Cannot apply Fiedler to more than 2 clusters!")

	# Compute all matrix. All of shape (n_samples, n_samples)

	# compute distances (sum of squared differences)
	dist_matrix = np.sum((np.expand_dims(data, 0) - np.expand_dims(data, 1)) ** 2, axis=2)

	# compute affinity (WEIGHT) matrix
	affinity_matrix = np.exp(- dist_matrix / (sigma ** 2))

	# compute degree matrix
	degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))

	# compute laplacian
	laplacian_matrix = degree_matrix - affinity_matrix

	# compute eigenvalues and vectors (suggestion: np.linalg is your friend)
	eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)

	# ensure we are not using complex numbers - you shouldn't btw
	if eigenvalues.dtype == 'complex128':
		print(
			"My dude, you got complex eigenvalues. Now I am not gonna break down,"
			"but you should totally give me higher sigmas (σ). (;")
		eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real

	# sort eigenvalues and vectors
	sorted_indices = np.argsort(eigenvalues)
	eigenvalues = eigenvalues[sorted_indices] # Has shape (n_samples,)
	eigenvectors = eigenvectors[:, sorted_indices] # Every column is a eigenvector. Has shape (n_samples, n_samples)

	# SOLUTION A: Fiedler-vector solution
	# - consider only the SECOND smallest eigenvector
	# - threshold it at zero
	# - return as labels
	labels = eigenvectors[:, 1] > 0 # Second smallest eigenvector as binary labels
	if fiedler_solution:
		return labels

	# SOLUTION B: K-Means solution
	# - consider eigenvectors up to the n_cl-th
	# - use them as features instead of data for KMeans
	# - You want to use sklearn's implementation (;
	# - return KMeans' clusters
	new_features = eigenvectors[:, 1:n_cl + 1] # We exclude the first eigenvector, which is not useful for clustering
	labels = KMeans(n_cl, n_init='auto').fit_predict(new_features)
	return labels


def main_spectral_clustering():
	"""
	Main function for spectral clustering.
	"""

	# generate the dataset
	data, cl = two_moon_dataset(n_samples=300, noise=0.1)  # best sigma = 0.1
	# data, cl = gaussians_dataset(n_gaussian=3, n_points=[100, 100, 70], mus=[[1, 1],
	# 								[-4, 6], [8, 8]], stds=[[1, 1], [3, 3], [1, 1]])  # best sigma = 2

	# visualize the dataset
	_, ax = plt.subplots(1, 2)
	ax[0].scatter(data[:, 0], data[:, 1], c=cl, s=40)

	# run spectral clustering
	labels = spectral_clustering(data, n_cl=2, sigma=0.1, fiedler_solution=True) # two moons
	#labels = spectral_clustering(data, n_cl=3, sigma=2, fiedler_solution=False) # gaussians

	# visualize results
	ax[1].scatter(data[:, 0], data[:, 1], c=labels, s=40)
	plt.waitforbuttonpress()


if __name__ == '__main__':
	main_spectral_clustering()
