from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import helper


def null_vector():
  # MISSING: Create a vector of size 10 and of type np.float32 full of zeros.
  return np.zeros((10))


def chess():
  # MISSING: Create a 8x8 matrix and fill it with a checkerboard pattern
  # starting with 1 in the upper left and finishing with 0.
  # The matrix type should be np.int32.
  chess = np.tile(np.concatenate((np.tile([1,0], 4), np.tile([0,1], 4))), 4)
  chess.shape = (8,8)
  return chess



def polar(a):
  # MISSING: "a" is a Nx2 matrix representing cartesian coordinates,
  # convert them to polar coordinates.
  # This function should return a Nx2 matrix where:
  # - the first column is the radius.
  #  -the second column is the angle.
  x = a[:,0].transpose()
  y = a[:,1].transpose()
  b = np.column_stack((np.hypot(x, y), np.arctan2(x,y)))
  return b


def cap(a, b):
  # MISSING: Cap all elements of "a" at "b" (i.e., max(a_{i,j}) <= b).
  # BONUS: Make sure to do so without modifying the input arguments.
  c = np.where(a <=b, a, b)
  return c


def moving_average(a):
  # MISSING: Compute the moving average of the elements of "a" over a
  # window of size 3.
  return np.convolve(a, np.ones(3), 'valid') / 3


def main():
  helper.check_solution(null_vector, [0] * 10)
  helper.check_solution(chess, [[1, 0, 1, 0, 1, 0, 1, 0],
                                [0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1, 0],
                                [0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1, 0],
                                [0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1, 0],
                                [0, 1, 0, 1, 0, 1, 0, 1]])
  helper.check_solution(polar, [[0.70710678118, np.pi / 4.],
                                [0.70710678118 * 2, np.pi / 4.]],
                        np.array([[.5, .5], [1., 1.]], np.float32),
                        approx=True)
  helper.check_solution(cap, [1, 2, 2], np.array([1, 2, 3]), 2)
  helper.check_solution(cap, [2, 1, 2], np.array([3., 1., 3.]), 2)
  helper.check_solution(moving_average, [2, 3], np.array([1., 2., 3., 4.]))
  helper.check_solution(moving_average, [2], np.array([2., 2., 2.]))


if __name__ == '__main__':
  main()
