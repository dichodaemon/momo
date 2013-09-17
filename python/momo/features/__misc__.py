import numpy as np

DIRECTIONS = np.array( [
  [ 1.,  0.],
  [ 1.,  1.],
  [ 0.,  1.],
  [-1.,  1.],
  [-1.,  0.],
  [-1., -1.],
  [ 0., -1.],
  [ 1., -1.]
], dtype = np.float32 )


def feature_sum( feature_module, states, frames, radius ):
  result = np.array( [0.] * feature_module.FEATURE_LENGTH )
  for i in xrange( len( states ) ):
    result += feature_module.compute_feature( states[i], frames[i], radius )
  return result

