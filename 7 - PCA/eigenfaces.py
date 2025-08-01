"""
Eigenfaces main script.
"""

import numpy as np

from utils import show_eigenfaces
from utils import show_nearest_neighbor
from data_io import get_faces_dataset

import matplotlib.pyplot as plt

plt.ion()


def main():
	# get_data
	X_train, Y_train, X_test, Y_test = get_faces_dataset(path='att_faces', train_split=0.6)

	# number of principal components
	n_components = 30

	# fit the PCA transform
	eigpca = Eigenfaces(n_components)
	eigpca.fit(X_train, verbose=True)

	# project the training data
	proj_train = eigpca.transform(X_train)

	# project the test data
	proj_test = eigpca.transform(X_test)

	# fit a 1-NN classifier on PCA features
	nn = NearestNeighbor()
	nn.fit(proj_train, Y_train)

	# Compute predictions and indices of 1-NN samples for the test set
	predictions, nearest_neighbors = nn.predict(proj_test)

	# Compute the accuracy on the test set
	test_set_accuracy = float(np.sum(predictions == Y_test)) / len(predictions)
	print(f'Test set accuracy: {test_set_accuracy}')

	# Show results.
	show_nearest_neighbor(X_train, Y_train, X_test, Y_test, nearest_neighbors)


class Eigenfaces:
	# Performs PCA to project faces in a reduced space.
	def __init__(self, n_components: int):
		# n_components: integer number of principal component
		self.n_components = n_components

		# Per-feature empirical mean, estimated from the training set.
		self._x_mean = None

		# Principal axes in feature space, representing the directions
		# of maximum variance in the data.
		self._ef = None

	def fit(self, X: np.ndarray, verbose: bool = False):
		"""
		Parameters:
			X: ndarray training set that will be used for fitting the PCA. Has shape: (n_samples, w * h)
			verbose: boolean flag
		"""

		# compute mean vector and store it for the inference stage
		self._x_mean = np.mean(X, axis=0)

		if verbose:
			# show mean face
			plt.imshow(np.reshape(self._x_mean, newshape=(112, 92)), cmap='gray')
			plt.title('mean face')
			plt.waitforbuttonpress(timeout=0.1)
			plt.close()

		# remove mean from the data
		X_norm = X - self._x_mean

		# compute covariance with the eigen-trick
		cov = X_norm @ X_norm.T

		# compute eigenvectors of the covariance matrix
		eigval, eigvec = np.linalg.eig(cov)

		# sort them (decreasing order w.r.t. eigenvalues)
		sorted_indices = np.argsort(eigval)[::-1]  # argsort is ascending order but we want descending, so we use [::-1]
		sorted_eigval = eigval[sorted_indices]
		sorted_eigvec = eigvec[:, sorted_indices]

		# select principal components (vectors)
		principal_sorted_eigval = sorted_eigval[:self.n_components]
		principal_sorted_eigvec = sorted_eigvec[:, :self.n_components]

		# retrieve original eigenvec
		self._ef = X.T @ principal_sorted_eigvec

		# normalize the retrieved eigenvectors to have unit length
		self._ef /= np.sqrt(principal_sorted_eigval)

		if verbose:
			# show eigenfaces
			show_eigenfaces(self._ef, (112, 92))
			plt.waitforbuttonpress(timeout=0.1)
			plt.close()

	def transform(self, X: np.ndarray):
		"""
		Parameters:
			X: ndarray of faces to project. Has shape: (n_samples, w*h)
		Returns an ndarray of projections. Has shape: (n_samples, self.n_components)
		"""

		# project faces according to the computed directions
		return (X - self._x_mean) @ self._ef

	def inverse_transform(self, X: np.ndarray):
		return (X @ self._ef.T) + self._x_mean


class NearestNeighbor:

	def __init__(self):
		self._X_db, self._Y_db = None, None

	def fit(self, X: np.ndarray, y: np.ndarray, ):
		# Fit the model using X as training data and y as target values
		self._X_db = X
		self._Y_db = y

	def predict(self, X: np.ndarray):
		# Finds the 1-neighbor of a point. Returns predictions as well as indices of the neighbors of each point.
		num_test_samples = X.shape[0]

		# predict test faces
		predictions = np.zeros((num_test_samples,))
		nearest_neighbors = np.zeros((num_test_samples,), dtype=np.int32)

		for i in range(num_test_samples):
			distances = np.sum(np.square(self._X_db - X[i]), axis=1)

			# nearest neighbor classification
			nearest_neighbor = np.argmin(distances)
			nearest_neighbors[i] = nearest_neighbor
			predictions[i] = self._Y_db[nearest_neighbor]

		return predictions, nearest_neighbors


if __name__ == '__main__':
	main()
