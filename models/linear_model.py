# Linear models
# Initialization automatically fits
from abc import ABC, abstractmethod
import numpy as np

class LinearModel(ABC):
  @abstractmethod
  def fit(X:np.ndarray, y:np.ndarray):
    pass

  @abstractmethod
  def predict(X:np.ndarray) -> np.ndarray:
    pass
  

class LinearRegression(LinearModel):
  def __init__(self):
    self.fitted_ = False
    self.coef_ = 0
    self.intercept_ = 0

  def fit(self, X, y):
    n, k = X.shape
    X = np.hstack((np.ones([X.shape[0], 1]), X))
    betas = np.linalg.inv(X.T @ X) @ X.T @ y
    self.fitted_ = True
    self.intercept_ = betas[0]
    self.coef_ = betas[1:]
    return self

  def predict(self, X):
    X = np.hstack((np.ones([X.shape[0], 1]), X))
    return X @ np.insert(self.coef_, 0, self.intercept_, axis=0)
