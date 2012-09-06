import pyopencl as cl
import numpy as np
from math import *

def loadProgram( context, filename ):
  f = open( filename, 'r' )
  fstr = "".join( f.readlines() )
  program = cl.Program( context, fstr ).build()
  return program

class CL:
  def __init__(self):
    self.context = cl.create_some_context()
    self.queue = cl.CommandQueue( self.context )
    self.program = loadProgram( self.context, "computeFeature.cl" )

    mf = cl.mem_flags
    actions = np.array( [
      [ 1.,  0.],
      [ 1.,  1.],
      [ 0.,  1.],
      [-1.,  1.],
      [-1.,  0.],
      [-1., -1.],
      [ 0., -1.],
      [ 1., -1.]
    ], dtype = np.float32 )
    angles = np.array( [
      [cos( pi / 8 ), sin( pi / 8 )],
      [cos( 3 * pi / 8 ), sin( 3 * pi / 8 )],
      [cos( 5 * pi / 8 ), sin( 5 * pi / 8 )],
      [cos( 7 * pi / 8 ), sin( 7 * pi / 8 )],
    ], dtype = np.float32 )
    speeds = np.array( [0., 0.02, 0.05], dtype = np.float32 )
    self.action_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = actions )
    self.angle_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = angles )
    self.speed_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = speeds )

  def compute_weights( self, grid_width, grid_height, delta, speed, theta, frame ):
    mf = cl.mem_flags
    costs = np.zeros( (8, grid_height, grid_width), dtype=np.float32 )
    theta_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = theta  )
    frame_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = frame  )
    cost_buffer  = cl.Buffer( self.context, mf.WRITE_ONLY, costs.nbytes )
    self.program.computeWeights( 
      self.queue, costs.shape, None, 
      np.float32( speed ), np.float32( delta ), 
      np.int32( grid_width ), np.int32( grid_height ),
      np.int32( frame.shape[0] ), frame_buffer, 
      self.action_buffer, self.angle_buffer, self.speed_buffer,      
      theta_buffer, cost_buffer 
    )
    cl.enqueue_read_buffer( self.queue, cost_buffer, costs ).wait()
    return costs

if __name__ == "__main__":
  import time
  import pylab as pl
  frame  = np.array( [ 
    [1, 2, .021, 0.]
  ], dtype=np.float32 )
  theta = np.random.rand( 17 ).astype( np.float32 )
  theta /= np.linalg.norm( theta )
  example = CL()
  width  = 512
  height = 256
  delta  = 0.05
  result = example.compute_weights( width, height, delta, 0.022, theta, frame )
  pl.figure( 1, figsize = ( 10, 10 ), dpi = 75 )
  pl.imshow( result[0], pl.cm.jet, None, None, "none", extent = (0, width * delta, 0, height * delta ), origin = "lower" )
  pl.show()

