# Linear models
# Initialization automatically fits
from abc import ABC, abstractmethod
import numpy as np

class LinearModel(ABC):
  @abstractmethod
  def fit(X:np.ndarray, y:np.ndarray):
    pass

  @abstractmethod
  def pred(X:np.ndarray) -> np.ndarray:
    pass
  

class LinearRegression(LinearModel):
  def __init__(self):
    fitted_ = False
    coef_ = 0
    intercept_ = 0
  def fit(self, X, y):
    n, k = X.shape
    X = np.hstack((np.ones([X.shape[0], 1]), X))
    betas = np.linalg.inv(X.T @ X) @ X.T @ y
    self.fitted_ = True
    self.intercept_ = betas[0]
    self.coef_ = betas[1:]
    return self
  def pred(self, X):
    pass
