import numpy as np
from momo import accum

def mlirl( mdp, data ):
  counts = accum( data, 1 )
  policy = counts / np.transpose( np.tile( np.sum( counts, 1 ), ( mdp.num_actions, 1 ) ) )
  q = 1. / 10 * np.log( counts )
  q_max = np.max( q, 1 )
  return ml_q - mdp.discount * np.transpose( np.dot( mdp.transition, q_max ) )
