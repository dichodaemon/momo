import pyopencl as cl
import numpy as np
from math import *
import momo
from momo.features import *

class compute_costs( momo.opencl.Program ):
  def __init__( self, convert ):
    momo.opencl.Program.__init__( self )
    self.flow = self.loadProgram( momo.BASE_DIR + "/opencl/computeCosts.cl" )

    self.convert = convert

    mf = cl.mem_flags
    self.direction_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = DIRECTIONS )

  def __call__( self, features, theta ):
    mf = cl.mem_flags
    costs = np.zeros( (8, self.convert.grid_height, self.convert.grid_width ), dtype=np.float64 )
    feature_length = features.shape[3]
    
    features_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = features  )
    theta_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = theta.astype( np.float64 )  )
    cost_buffer  = cl.Buffer( self.context, mf.WRITE_ONLY, costs.nbytes )

    self.flow.computeCosts( 
      self.queue, costs.shape, None, 
      np.int32( self.convert.grid_width ), np.int32( self.convert.grid_height ), np.int32( feature_length ),
      self.direction_buffer, 
      features_buffer,      
      theta_buffer, cost_buffer 
    )

    cl.enqueue_copy( self.queue, costs, cost_buffer )
    return costs

