import numpy as np
import copy
import matplotlib.pylab as pl
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
import sys
#import util
import time as comp_timer
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
from scipy.special import expit
from scipy.linalg import solve_triangular

class SBHM(LinearClassifierMixin, BaseEstimator):
    def __init__(self, gamma=0.075*0.814, grid=None, cell_resolution=(5, 5), cell_max_min=None, X=None, calc_loss=False,fresh=True):
        """
        :param gamma: RBF bandwidth
        :param grid: if there are prespecified locations to hinge the RBF
        :param cell_resolution: if 'grid' is 'None', resolution to hinge RBFs
        :param cell_max_min: if 'grid' is 'None', realm of the RBF field
        :param X: a sample of lidar locations to use when both 'grid' and 'cell_max_min' are 'None'
        """
        self.gamma = gamma
        self.fresh = fresh
        if grid is not None:
            self.grid = grid
        else:
            self.grid = self.__calc_grid_auto(cell_resolution, cell_max_min, X)
        self.calc_loss = calc_loss
        self.intercept_, self.coef_, self.sigma_ = [0], [0], [0]
        self.scan_no = 0
        print('D=', self.grid.shape[0])

    def __calc_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """

        if max_min is None:
            # if 'max_min' is not given, make a boundarary based on X
            # assume 'X' contains samples from the entire area
            expansion_coef = 1.2
            x_min, x_max = expansion_coef*X[:, 0].min(), expansion_coef*X[:, 0].max()
            y_min, y_max = expansion_coef*X[:, 1].min(), expansion_coef*X[:, 1].max()
        else:
            x_min, x_max = max_min[0], max_min[1]
            y_min, y_max = max_min[2], max_min[3]

        xx, yy = np.meshgrid(np.arange(x_min, x_max, cell_resolution[0]), \
                             np.arange(y_min, y_max, cell_resolution[1]))
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))

        return grid

    def __sparse_features(self, X):
        """
        :param X: inputs of size (N,2)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        rbf_features = rbf_kernel(X, self.grid, gamma=self.gamma)
        return rbf_features

    def __lambda(self, epsilon):
        """
        :param epsilon: epsilon value for each data point
        :return: local approximation
        """
        return 0.5/epsilon*(expit(epsilon)-0.5)

    def __calc_loss(self, X, mu0, Sig0_inv, mu, Sig_inv, Ri, epsilon):
        Sig = Ri
        Sig0_inv=np.diag(np.diag(Sig0_inv))
        S0 = np.diag(np.divide(1,np.diag(Sig0_inv)))
        loss = 0.5 * np.linalg.slogdet(Sig)[1] - 0.5 * np.linalg.slogdet(S0)[1] + 0.5 * mu.T.dot(Sig_inv.dot(mu)) - 0.5 * mu0.T.dot(Sig0_inv.dot(mu0))
        loss += (np.sum(np.log(expit(epsilon)) - 0.5 * epsilon + self.__lambda(epsilon) * epsilon ** 2))

        return loss

    def __calc_posterior(self, X, y, epsilon, mu0, Sig0_inv, full_covar=False):

        lam = self.__lambda(epsilon)

        Sig_inv=np.zeros(Sig0_inv.shape)	

        for i in range(0,X.shape[1]):
                Sig_inv[i,i]=Sig0_inv[i,i]+2*np.dot(lam*X[:,i],X.T[i,:])
	
        #Sig_inv = np.diag(np.diag(Sig0_inv + 2 * np.dot(X.T*lam, X)))
	
        Sig=np.diag(np.divide(1,np.diag(Sig_inv)))
	
	#mu = np.zeros((mu0.shape))
        term=np.dot(X.T, (y - 0.5))
        part1=Sig0_inv.dot(mu0)
        mu=np.dot(Sig, part1 + term)
	#for i in range(0,y.shape[0]):
		#mu[i]=np.dot(Sig[i,i],(np.dot(Sig0_inv[i,i],mu0[i])+term[i]))

		
	
        if full_covar:
            return mu, Sig
        else:
            return mu, Sig_inv, Sig

    def fit(self, X, y):
        # If first run, set m0, S0i
        if self.fresh==True:
            self.mu = np.zeros((self.grid.shape[0]))
            self.Sig_inv = 0.0001 * np.eye((self.grid.shape[0])) #0.01 for sim, 0
            self.n_iter = 3
            self.fresh=False
        else:
            self.n_iter = 1

        epsilon = 1
        X_orig = copy.copy(X)

        for i in range(self.n_iter):
            X = self.__sparse_features(X_orig)

            # E-step: update Q(w)
            self.mu, self.Sig_inv, self.sig = self.__calc_posterior(X, y, epsilon, self.mu, self.Sig_inv)

            # M-step: update epsilon
            XMX = np.dot(X, self.mu)**2
            XSX = np.sum(np.dot(np.dot(X,self.sig),X.T),axis=1)
            epsilon = np.sqrt(XMX + XSX)

            # Calculate loss, if specified
            if self.calc_loss is True:
                print("  scan={}, iter={} => loss={:.1f}".format(self.scan_no, i,
                      self.__calc_loss(X, np.zeros((self.grid.shape[0])), 0.01*np.eye((self.grid.shape[0])),
                        self.mu, self.Sig_inv, self.Ri, epsilon)))

        self.intercept_ = [0]
        coef_, sigma_ = self.__calc_posterior(X, y, epsilon, self.mu, self.Sig_inv, True)

        self.intercept_ = 0
        self.coef_[0] = coef_
        self.sigma_[0] = sigma_
        self.coef_ = np.asarray(self.coef_)
        self.scan_no += 1

    def predict_proba(self, X_q):
        X_q = self.__sparse_features(X_q)#[:, 1:]
        scores = self.decision_function(X_q)

        sigma = np.asarray([np.sum(np.dot(X_q, s) * X_q, axis=1) for s in self.sigma_])
        ks = 1. / (1. + np.pi * sigma / 8) ** 0.5
        probs = expit(scores.T * ks).T
        if probs.shape[1] == 1:
            probs = np.hstack([1 - probs, probs])
        else:
            probs /= np.reshape(np.sum(probs, axis=1), (probs.shape[0], 1))
        return probs


