"""
Implementations of linear models. Closely follows the sklearn API.
"""

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

    
class LogisticRegression(LinearModel):
  def __init__(self):
    self.fitted_ = False
    self.coef_ = 0
    self.intercept_ = 0
  
  # Fit using stochastic gradient descent for now. Assume y = 0 or y = 1
  def fit(self, X, y):
  """
  z = w @ x + b    # logit. Note that logit(p) = sigmoid_inv(p) = ln (p / 1-p)
  sigmoid(z) = 1 / (1 + exp(-z))
  P(y = 1) = sigmoid(z)    
  P(y = 0) = 1 - sigmoid(z) = sigmoid(-z)
  cross entropy loss: -1/n \sum_{i=1}^m y_i log(hat y) + (1-y_i) ln (1-y_i)

  Gradients: wrt w: 1/n (z - y) \cdot x
            wrt b:  1/n (z - y)
  """


