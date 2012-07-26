import numpy as np
import matplotlib.pylab as pl
import scipy.ndimage as ni

class full_plan( object ):
  def __init__( self, cost_function, x, y, width, height, delta ):
    self.cost_function = cost_function
    self.x = x
    self.y = y
    self.height = height
    self.width = width
    self.delta = delta
    self.grid = np.zeros( ( self.height / self.delta, self.width / self.delta ) )

  def update_grid( self, frame ):
    y = self.y
    for i in xrange( self.grid.shape[0] ):
      x = self.x
      for j in xrange( self.grid.shape[1] ):
        reference = np.array( [x, y] )
        self.grid[i, j] = self.cost_function( reference, frame )
        x += self.delta
      y -= self.delta

  def neighbors( self, cummulated, phi, point ):
    for i in xrange( point[0] - 1, point[0] + 2 ):
      for j in xrange( point[1] -1, point[1] + 2 ):
        if i != point[0] or j != point[1]:
          if (     i >= 0 and i < self.grid.shape[0] 
               and j >= 0 and j < self.grid.shape[1] ):
            di = i - point[0]
            dj = j - point[1]
            d = ( di**2 + dj**2 )**0.5
            cost = cummulated[point[0], point[1]] + d * self.grid[i, j] 
            if cost < cummulated[i, j]:
              cummulated[i, j] = cost
              phi[i][j] = point 
              yield [i, j]

  def compute_policy( self, goal ):
    phi = [ [-1] * self.grid.shape[1] for i in xrange( self.grid.shape[0] )]
    cummulated = self.grid * 0 + 1E6

    print "Goal(1):", goal

    goal = [int( ( self.y - goal[1] ) / self.delta ), int( ( goal[0] - self.x ) / self.delta )]
    print "Goal(2):", goal
    cummulated[goal[0], goal[1]] = 0

    queue = [goal[0] * self.grid.shape[1] + goal[1]]

    while len( queue ) > 0:
      current = queue.pop( 0 )
      current_point = [current / self.grid.shape[1], current % self.grid.shape[1]]
      for n_point in self.neighbors( cummulated, phi, current_point ):
        n = n_point[0] * self.grid.shape[1] + n_point[1]
        if not n in queue:
          queue.append( n )
      queue.sort( key = lambda point: cummulated[point / self.grid.shape[1], point % self.grid.shape[1]] )
    return cummulated, phi

  def compute_gradient( self, cummulated ):
    x, y = pl.meshgrid( 
      pl.linspace( self.x, self.x + self.width, self.width / self.delta, endpoint = False ), 
      pl.linspace( self.y, self.y - self.height, self.height / self.delta, endpoint = False )
    )
    imx = pl.zeros( cummulated.shape )
    imy = pl.zeros( cummulated.shape )
    ni.filters.sobel( cummulated, 1, imx )
    ni.filters.sobel( cummulated, 0, imy )
    imx = -imx
    return x, y, imx, imy

  def descend_gradient( self, start, distance, imx, imy, step = 0.05 ):
    point = start * 1.
    while True:
      coords = pl.array( [int( ( point[0] - self.x ) / self.delta ), int( ( self.y - point[1] ) / self.delta ) ] )
      angle = pl.array( [imx[coords[1], coords[0]], imy[coords[1], coords[0] ] ] )
      angle = angle / ( angle[0]**2 + angle[1]**2 )**0.5
      if distance > step:
        point = point + angle * step
      else:
        point = point + angle * distance
        break
      distance -= step
    return point

  def __call__( self, goal, current, frame, distance ):
    self.update_grid( frame )
    cummulated, phi = self.compute_policy( goal )
    x, y, imx, imy = self.compute_gradient( cummulated )
    next_point = self.descend_gradient( current, distance, imx, imy )
    return next_point, x, y, imx, imy

