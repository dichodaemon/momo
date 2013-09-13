import numpy as np

def feature_sum( feature_module, states, frames, radius ):
  result = np.array( [0.] * feature_module.FEATURE_LENGTH )
  for i in xrange( len( states ) ):
    result += feature_module.compute_feature( states[i], frames[i], radius )
  return result

