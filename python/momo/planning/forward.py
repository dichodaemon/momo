import pyopencl as cl
import numpy as np
from math import *
from __common__ import *
import momo

class forward( momo.opencl.Program ):
  def __init__( self, convert, radius ):
    momo.opencl.Program.__init__( self )
    self.flow = self.loadProgram( momo.BASE_DIR + "/opencl/forward.cl" )

    self.convert = convert

    mf = cl.mem_flags

    self.idirection_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = momo.irl.features.flow.DIRECTIONS.astype( np.int32 ) )

  def __call__( self, costs, start, goal ):
    if ( costs < 0 ).any():
      raise runtime_error( "The cost matrix cannot have negative values" )
    mf = cl.mem_flags

    width = costs.shape[2]
    height = costs.shape[1]


    floats = np.zeros( costs.shape, dtype=np.float32 )
    ints   = np.zeros( costs.shape, dtype=np.int32 )
    ints[] = 1

    start_buffer  = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = start.astype( np.int32 ) )
    goal_buffer  = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = goal.astype( np.int32 ) )
    cost_buffer  = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = costs.astype( np.float32 ) )
    mask_buffer  = cl.Buffer( self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = ints )
