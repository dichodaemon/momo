import pyopencl as cl
import numpy as np
from math import *
from __common__ import *
import momo

class dijkstra( momo.opencl.Program ):
  def __init__( self ):
    momo.opencl.Program.__init__( self )
    self.dijkstra = self.loadProgram( momo.BASE_DIR + "/opencl/dijkstra.cl" )

    mf = cl.mem_flags
    self.idirection_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = DIRECTIONS )

  def __call__( self, costs, goal ):
    if ( costs < 0 ).any():
      raise runtime_error( "The cost matrix cannot have negative values" )
    mf = cl.mem_flags

    width = costs.shape[2]
    height = costs.shape[1]


    floats = np.zeros( costs.shape, dtype=np.float32 )
    ints   = np.zeros( costs.shape, dtype=np.int32 )

    goal_buffer  = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = goal.astype( np.int32 ) )
    cost_buffer  = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = costs.astype( np.float32 ) )

    cummulated_buffer  = cl.Buffer( self.context, mf.READ_WRITE, floats.nbytes )
    tcummulated_buffer  = cl.Buffer( self.context, mf.READ_WRITE, floats.nbytes ) 
    parent_buffer  = cl.Buffer( self.context, mf.READ_WRITE, ints.nbytes )
    tparent_buffer  = cl.Buffer( self.context, mf.READ_WRITE, ints.nbytes )
    mask_buffer  = cl.Buffer( self.context, mf.READ_WRITE, ints.nbytes )

    e1 = self.dijkstra.initializeBuffers( 
      self.queue, costs.shape, None, 
      np.int32( width ), np.int32( height ),
      mask_buffer, cummulated_buffer, tcummulated_buffer, 
      parent_buffer, tparent_buffer, 
      goal_buffer
    )

    count = 0
    while True:
      self.dijkstra.dijkstraPass1( 
        self.queue, costs.shape, None, 
        np.int32( width ), np.int32( height ),
        self.idirection_buffer, cost_buffer,
        mask_buffer, cummulated_buffer, tcummulated_buffer, 
        tparent_buffer, wait_for = [e1] 
      )

      self.dijkstra.dijkstraPass2( 
        self.queue, costs.shape, None, 
        np.int32( width ), np.int32( height ),
        mask_buffer, cummulated_buffer, tcummulated_buffer, 
        parent_buffer, tparent_buffer
      )
      count += 1
      if count % 40 == 0:
        cl.enqueue_copy( self.queue, ints, mask_buffer )
        if not ints.any():
          break

    cl.enqueue_copy( self.queue, floats, cummulated_buffer )
    cl.enqueue_copy( self.queue, ints, parent_buffer )

    return floats, ints

  def get_path( self, parents, start ):
    width = parents.shape[2]
    height = parents.shape[1]
    p1 = start[2] * width * height + start[1] * width + start[0]
    result = []
    while p1 != -1:
      k = p1 / ( height * width )
      y = ( p1 % ( height * width ) ) / width
      x = p1 % width
      result.append( [x, y, k] )
      p1 = parents[k, y, x]
    return np.array( result, dtype = np.int32 )
