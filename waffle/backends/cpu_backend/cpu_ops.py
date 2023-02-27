import numpy as np


def neg(x:np.ndarray) -> np.ndarray:
  return 0.0-x

def relu(x:np.ndarray) -> np.ndarray:
  return np.maximum(x, 0)

def exp(x:np.ndarray) -> np.ndarray:
  return np.exp(x)

def log(x:np.ndarray) -> np.ndarray:
  return np.log(x)

def reciprocal(x:np.ndarray) -> np.ndarray:
  return 1.0/x

def add(x:np.ndarray, y:np.ndarray) -> np.ndarray:
  return x+y

def sub(x:np.ndarray, y:np.ndarray) -> np.ndarray:
  return x-y

def mul(x:np.ndarray, y:np.ndarray) -> np.ndarray:
  return x*y

def div(x:np.ndarray, y:np.ndarray) -> np.ndarray:
  return x/y

def pow(x:np.ndarray, y:np.ndarray) -> np.ndarray:
  return x**y

def sum(x:np.ndarray, axis=None) -> np.ndarray:
  return np.sum(x, axis)

def max(x:np.ndarray, axis=None) -> np.ndarray:
  return x.max(axis)
