import pyopencl as cl
import numpy as np
from math import *
import momo

class compute_cummulated( momo.opencl.Program ):
  def __init__( self ):
    momo.opencl.Program.__init__( self )
    self.henryCummulated = self.loadProgram( momo.BASE_DIR + "/opencl/henryCummulated.cl" )

    mf = cl.mem_flags

    self.idirection_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = momo.planning.DIRECTIONS )


  def __call__( self, forward, backward, costs, features, origin, h ):
    mf = cl.mem_flags

    w_features = features.astype( np.float64 )
    cummulated = costs * 1

    forward_buffer  = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = forward )
    backward_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = backward )
    cost_buffer     = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = costs )
    cumm_buffer     = cl.Buffer( self.context, mf.READ_WRITE, costs.nbytes )
    feature_buffer  = cl.Buffer( self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = w_features )
    origin_buffer   = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = origin )

    self.henryCummulated.cummulated1( 
      self.queue, costs.shape, None, 
      np.int32( w_features.shape[2] ), np.int32( w_features.shape[1] ),
      origin_buffer, np.int32( h ),
      self.idirection_buffer,
      forward_buffer, backward_buffer, cost_buffer, cumm_buffer
    )

    self.henryCummulated.cummulated2( 
      self.queue, costs.shape, None, 
      np.int32( w_features.shape[2] ), np.int32( w_features.shape[1] ),
      np.int32( features.shape[3] ),
      cumm_buffer, feature_buffer
    )

    cl.enqueue_copy( self.queue, cummulated, cumm_buffer )
    cl.enqueue_copy( self.queue, w_features, feature_buffer )
    return cummulated, w_features

