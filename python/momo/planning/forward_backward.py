import sys
import os

BASE_DIR = os.path.abspath( os.path.join( os.path.dirname( __file__ ), "../../.." ) )
path     = os.path.abspath( os.path.join( BASE_DIR, "python" ) )
sys.path.append( path )

import pyopencl as cl
import numpy as np
import pylab as pl
from math import *
import momo
from __common__ import *
import time

class forward_backward( momo.opencl.Program ):
  def __init__( self ):
    momo.opencl.Program.__init__( self )
    self.flow = self.loadProgram( momo.BASE_DIR + "/opencl/forward_backward.cl" )

    mf = cl.mem_flags

    self.idirection_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = DIRECTIONS )

  def __call__( self, costs, start, sign ):
    if ( costs < 0 ).any():
      raise runtime_error( "The cost matrix cannot have negative values" )
    mf = cl.mem_flags

    width = costs.shape[2]
    height = costs.shape[1]


    floats = np.zeros( costs.shape, dtype=np.float32 )
    ints   = np.zeros( costs.shape, dtype=np.int32 )
    ints[tuple( reversed( start.tolist() ) )] = 1

    cost_buffer  = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = costs.astype( np.float32 ) )
    mask_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = ints )
    f1_buffer  = cl.Buffer( self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = floats )
    f2_buffer  = cl.Buffer( self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = floats )
    

    for i in xrange( sum( costs.shape ) ):
      if sign > 0:
        e1 = self.flow.forwardPass( 
          self.queue, costs.shape, None, 
          np.int32( width ), np.int32( height ),
          self.idirection_buffer, cost_buffer,
          mask_buffer,  
          f1_buffer, f2_buffer
        )
      else:
        e1 = self.flow.backwardPass( 
          self.queue, costs.shape, None, 
          np.int32( width ), np.int32( height ),
          self.idirection_buffer, cost_buffer,
          mask_buffer,  
          f1_buffer, f2_buffer
        )
      e2 = self.flow.updatePass( 
        self.queue, costs.shape, None, 
        np.int32( width ), np.int32( height ),
        f1_buffer, f2_buffer,
        wait_for = [e1] 
      )
      e2.wait()
    cl.enqueue_copy( self.queue, floats, f1_buffer )
    pl.ion()
    pl.clf()
    pl.imshow( np.sum( floats, 0 ) )
    pl.draw()
    return floats


if __name__ == "__main__":
  costs = np.ones( (8, 7, 8 ) )
  for i in xrange( 4 ):
    costs[i] *= ( i + 1 ) * 0.1
    if i != 0:
      costs[8 - i] *= ( i + 1 ) * 0.1

  start = np.array( [1, 3, 0], dtype = np.int32 )
  end = np.array( [6, 3, 0], dtype = np.int32 )

  fb = forward_backward()

  f = fb( costs, start, 1 )
  b = fb( costs, end, -1 )
  print b[0, 3, 1], f[6, 3, 0]

