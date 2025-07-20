from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform


class KMeans:

    def __init__(self, n_cl: int, n_init: int = 1,
                 initial_centers: Optional[np.ndarray] = None,
                 verbose: bool = False) -> None:
        """
        Parameters
        ----------
        n_cl: int
            number of clusters.
        n_init: int
            number of time the k-means algorithm will be run.
        initial_centers:
            If an ndarray is passed, it should be of shape (n_clusters, n_features)
            and gives the initial centers.
        verbose: bool
            whether or not to plot assignment at each iteration (default is True).
        """

        self.n_cl = n_cl
        self.n_init = n_init
        self.initial_centers = initial_centers
        self.verbose = verbose

    def _init_centers(self, X: np.ndarray, use_samples: bool = False):

        n_samples, dim = X.shape

        if use_samples:
            return X[np.random.choice(n_samples, size=self.n_cl)]

        centers = np.zeros((self.n_cl, dim))
        for i in range(dim):
            min_f, max_f = np.min(X[:, i]), np.max(X[:, i])
            centers[:, i] = uniform(low=min_f, high=max_f, size=self.n_cl)
        return centers

    def single_fit_predict(self, X: np.ndarray):
        """
        Kmeans algorithm.

        Parameters
        ----------
        X: ndarray
            data to partition, Expected shape (n_samples, dim).

        Returns
        -------
        centers: ndarray
            computed centers. Expected shape (n_cl, dim)

        assignments: ndarray
            assigned center between 0 and self.n_cl given to each point in X. Expected shape (n_samples)
        """

        n_samples, dim = X.shape

        # initialize centers
        centers = np.array(self._init_centers(X)) if self.initial_centers is None \
            else np.array(self.initial_centers)

        old_assignments = np.zeros(shape=n_samples)

        while True:  # stopping criterion

            # assign

            distances = np.zeros(shape=(n_samples, self.n_cl))
            for i in range(self.n_cl):
                distances[:, i] = np.sum(np.square(X - centers[i]), axis=-1)

            # Computing distances without for loop
            # distances = np.sum(np.square(X[:, None] - centers[None, :]), axis=-1)

            new_assignments = np.argmin(distances, axis=1)

            # re-estimate
            for l in range(0, self.n_cl):
                centers[l] = np.mean(X[new_assignments == l], axis=0)

            # break if no assignment has changed
            if np.all(new_assignments == old_assignments):
                break

            # update
            old_assignments = new_assignments

        return centers, new_assignments

    def compute_cost_function(self, X: np.ndarray, centers: np.ndarray,
                              assignments: np.ndarray):
        """
        Returns
        -------
        cost: float
            computed cost function.
        """

        cost = 0.0

        for i in range(self.n_cl):
            assigned_points = X[assignments == i]
            cost += np.sum(np.square(assigned_points - centers[i]))

        # without for loop
        # cost = np.sum(np.square(X[:, None] - centers[None, :])[np.arange(len(X)), assignments])

        return cost

    def fit_predict(self, X: np.ndarray):
        """
        Returns
        -------
        assignment: ndarray
            computed assignment. Expected shape (n_samples,)
        """
        assignments_opt = None
        cost_min = np.inf

        for i in range(self.n_init):
            centers, assignments = self.single_fit_predict(X)
            cost = self.compute_cost_function(X, centers, assignments)

            if self.verbose:
                print(f'Iteration: {i} - cost function: {cost_min: .2f}')
                fig, ax = plt.subplots()
                ax.scatter(X[:, 0], X[:, 1], c=assignments, s=40)
                ax.plot(centers[:, 0], centers[:, 1], 'r*', markersize=20)

            if cost < cost_min:
                cost_min = cost
                assignments_opt = assignments

        return assignments_opt
