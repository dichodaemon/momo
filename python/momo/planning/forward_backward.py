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
    m1_buffer  = cl.Buffer( self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = ints )
    m2_buffer  = cl.Buffer( self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = ints )
    f1_buffer  = cl.Buffer( self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = floats )
    f2_buffer  = cl.Buffer( self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = floats )
    
    pl.ion()

    for i in xrange( 300 ):
      e1 = self.flow.computeForward1( 
        self.queue, costs.shape, None, 
        np.int32( width ), np.int32( height ), np.int32( sign ),
        self.idirection_buffer, cost_buffer,
        m1_buffer, m2_buffer, 
        f1_buffer, f2_buffer
      )
      e2 = self.flow.computeForward2( 
        self.queue, costs.shape, None, 
        np.int32( width ), np.int32( height ),
        m1_buffer, m2_buffer, 
        f1_buffer, f2_buffer,
        wait_for = [e1] 
      )
      e2.wait()
      cl.enqueue_copy( self.queue, floats, f1_buffer )
      print i
      pl.clf()
      pl.imshow( floats[4] )
      pl.draw()
    cl.enqueue_copy( self.queue, floats, f1_buffer )
    return floats
