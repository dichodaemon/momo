import pyopencl as cl
import numpy as np
from math import *
from __common__ import *
import momo

class compute_costs( momo.opencl.Program ):
  def __init__( self, convert, radius ):
    momo.opencl.Program.__init__( self )
    self.flow = self.loadProgram( momo.BASE_DIR + "/opencl/flow.cl" )

    self.convert = convert
    self.radius = radius

    mf = cl.mem_flags
    self.fdirection_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = DIRECTIONS )
    self.angle_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = ANGLES )
    self.speed_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = SPEEDS )

  def __call__( self, speed, theta, frame ):
    mf = cl.mem_flags
    costs = np.zeros( (8, self.convert.grid_height, self.convert.grid_width ), dtype=np.float64 )

    theta_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = theta.astype( np.float64 )  )
    frame_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = self.convert.rebase_frame( frame ).astype( np.float32 ) )
    cost_buffer  = cl.Buffer( self.context, mf.WRITE_ONLY, costs.nbytes )

    self.flow.computeCosts( 
      self.queue, costs.shape, None, 
      np.float32( speed ), np.float32( self.convert.delta ), np.float32( self.radius ),
      np.int32( self.convert.grid_width ), np.int32( self.convert.grid_height ),
      np.int32( frame.shape[0] ), frame_buffer, 
      self.fdirection_buffer, self.angle_buffer, self.speed_buffer,      
      theta_buffer, cost_buffer 
    )

    cl.enqueue_copy( self.queue, costs, cost_buffer )
    return costs

