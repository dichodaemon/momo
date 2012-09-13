import momo

class dijkstra( object ):
  def __init__( self, convert, compute_costs ):
    self.convert = convert
    self.compute_costs = compute_costs
    self.planner = momo.planning.dijkstra()

  def __call__( self, start, goal, velocity, frames, replan, theta ):
    current = self.convert.from_world2( start )
    goal = self.convert.from_world2( goal )

    count = 0
    result = []

    while True:
      if count >= len( frames ):
        count = len( frames ) - 1

      costs = self.compute_costs( velocity, theta, frames[count] )
      cummulated, parents = self.planner( costs, goal )
      path = self.planner.get_path( parents, current )

      for p in path[:replan]:
        result.append( self.convert.to_world2( p, velocity ) )
      if  len( path ) == replan + 1:
        result.append( self.convert.to_world2( p, velocity ) )
      if len( path ) <= replan + 1:
        break

      current = path[replan]
    return result
