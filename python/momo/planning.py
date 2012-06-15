import matplotlib.pylab as pl
import numpy as np
import numbers

def neighbors( grid, cummulated, phi, point ):
  for i in xrange( point[0] - 1, point[0] + 2 ):
    for j in xrange( point[1] -1, point[1] + 2 ):
      if i != point[0] or j != point[1]:
        if (     i >= 0 and i < grid.shape[0] 
             and j >= 0 and j < grid.shape[1] ):
          di = i - point[0]
          dj = j - point[1]
          d = ( di**2 + dj**2 )**0.5
          cost = cummulated[point[0], point[1]] + d * grid[i, j] 
          if cost < cummulated[i, j]:
            cummulated[i, j] = cost
            phi[i][j] = point 
            yield [i, j]


def forward( grid, grid_x, grid_y, delta, goal ):
  phi = [ [-1] * grid.shape[1] for i in xrange( grid.shape[0] )]
  cummulated = grid * 0 + 1E6

  goal = [int( ( grid_y - goal[1] ) / delta ), int( ( goal[0] - grid_x ) / delta )]
  cummulated[goal[0], goal[1]] = 0

  queue = [goal[0] * grid.shape[1] + goal[1]]

  while len( queue ) > 0:
    current = queue.pop( 0 )
    current_point = [current / grid.shape[1], current % grid.shape[1]]
    for n_point in neighbors( grid, cummulated, phi, current_point ):
      n = n_point[0] * grid.shape[1] + n_point[1]
      if not n in queue:
        queue.append( n )
    queue.sort( key = lambda point: cummulated[point / grid.shape[1], point % grid.shape[1]] )
  return cummulated, phi
    
def get_path( phi, grid_x, grid_y, delta, start ):
  start = [int( ( grid_y - start[1] ) / delta ), int( ( start[0] - grid_x ) / delta )]
  path = []
  current = start
  while phi[current[0]][current[1]] != -1:
    path.append( np.array( [grid_x + current[1] * delta + delta / 2, grid_y - current[0] * delta - delta / 2] ) )
    current = phi[current[0]][current[1]]
  return np.array( path )

