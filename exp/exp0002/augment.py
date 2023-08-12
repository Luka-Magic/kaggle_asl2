import numpy as np
import math


class AffineMatTools():
    def __init__(self):
        self.A_list = []

    def composit(self):
        M = np.eye(3)
        for A in self.A_list:
            M = A.dot(M)
        M = M[:2]
        return M

    def composit3(self):
        M = np.eye(3)
        for A in self.A_list:
            M = A.dot(M)
        return M

    def adjust_composit(self, w, h):
        M = self.composit()
        X = np.array(((0, 0, 1), (w - 1, 0, 1),
                     (0, h - 1, 1), (w - 1, h - 1, 1))).T
        Z = M.dot(X)
        min_x, min_y = Z.min(axis=1)
        max_x, max_y = Z.max(axis=1)
        # print(max_x , min_x)
        # print(max_y , min_y)
        new_w = round(max_x - min_x + 1)
        new_h = round(max_y - min_y + 1)
        self.shift(-min_x, -min_y)
        M = self.composit()
        return M, new_w, new_h

    def scale(self, x, y=None):
        if y is None:
            y = x
        A = np.eye(3)
        A[0, 0] = x
        A[1, 1] = y
        self.A_list.append(A)

    def shift(self, x, y):
        A = np.eye(3)
        A[0, 2] = x
        A[1, 2] = y
        self.A_list.append(A)

    def rotation_radian(self, r):
        A = np.eye(3)
        A[0, 0] = math.cos(r)
        A[1, 1] = math.cos(r)
        A[0, 1] = -math.sin(r)
        A[1, 0] = math.sin(r)
        self.A_list.append(A)

    def rotation_degree(self, d):
        r = d * (2 * math.pi) / 360
        self.rotation_radian(r)

    def skew_x_radian(self, r):
        A = np.eye(3)
        A[0, 1] = math.tan(r)
        self.A_list.append(A)

    def skew_y_radian(self, r):
        A = np.eye(3)
        A[1, 0] = math.tan(r)
        self.A_list.append(A)

    def skew_x_degree(self, d):
        r = d * (2 * math.pi) / 360
        self.skew_x_radian(r)

    def skew_y_degree(self, d):
        r = d * (2 * math.pi) / 360
        self.skew_y_radian(r)

    def transform(self, data):
        data = data.copy()
        shape = data.shape
        data = data.reshape((-1, 2))
        data = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)
        M = self.composit3()
        data = data @ M.T
        data = (data / data[:, 2][:, None])[:, :2]
        data = data.reshape(shape)
        return data
