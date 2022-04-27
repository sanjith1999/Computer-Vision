import cv2 as cv
import numpy as np


def least_square_fitting(x, y):
    """
    Least Square Fitting
    -------------------
    x : x-co-ordinates (nd-array)
    y : y-co-ordinates (nd-array)
    return : m_star, c_star
    """
    n = len(x)
    X = np.concatenate([x.reshape(n, 1), np.ones((n, 1))], axis=1)
    B = np.linalg.pinv(X.T @ X) @ X.T @ y
    m_star = B[0]
    c_star = B[1]
    return m_star, c_star


def least_square_fitting_(x, y):
    """
    Least Square Fitting
    -------------------
    x : x-co-ordinates (nd-array)
    y : y-co-ordinates (nd-array)
    return : m_star, c_star
    """
    n = len(x)
    u11 = np.sum((x - np.mean(x)) ** 2)
    u12 = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    u21 = u12
    u22 = np.sum((y - np.mean(y)) ** 2)

    U = np.array([[u11, u12], [u21, u22]])
    W, V = np.linalg.eig(U)
    ev_corresponding_to_smallest_ev = V[:, np.argmin(W)]

    [a, b] = ev_corresponding_to_smallest_ev
    d = a * np.mean(x) + b * np.mean(y)

    m_star = -a / b
    c_star = d / b

    return m_star, c_star
