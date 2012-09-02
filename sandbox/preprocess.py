
  def preprocess( self, data ):
    min_x = 0.
    min_y = 0.
    max_x = 0.
    max_y = 0.
    frame_data = {}
    for frame in momo.frames( data ):
      tmp = []
      for o_frame, o_time, o_id, o_x, o_y, o_dx, o_dy in frame:
        if o_x < min_x:
          min_x = o_x
        if o_y < min_y:
          min_y = o_y
        if o_x > max_x:
          max_x = o_x
        if o_y > max_y:
          max_y = o_y
        tmp.append( [o_id, np.array( [o_x, o_y, o_dx, o_dy] )] )
      for i in xrange( len( frame ) ):
        o_id, o = tmp.pop( 0 )
        feature = self.module.compute( o, o, [t[1] for t in tmp] )
        if not o_id in frame_data:
          frame_data[o_id] = {
            "features": [],
            "states": [], 
            "frames": []
          }
        frame_data[o_id]["states"].append( o )
        frame_data[o_id]["features"].append( feature )
        frame_data[o_id]["frames"].append( [f[1] for f in tmp] )
        tmp.append( [o_id, o] )
    self.x = min_x
    self.y = min_y
    self.width = ( max_x - min_x ) * 1.01
    self.height = ( max_y - min_y ) * 1.01
    if self.width > self.height:
      self.delta  = self.width / 64
    else:
      self.delta = self.height / 64

    self.grid = momo.planning.grid( 
      self.x, self.y, self.width, self.height, self.delta
    )

    self.width = self.grid.width
    self.height = self.grid.height

    print "*" * 80
    print "%i paths to process" % len( frame_data.items() )
    print "Upper corner: (%f, %f)" % ( self.x, self.y )
    print "Dimensions: (%f, %f)" % ( self.width, self.height )
    print "Grid dimensions: (%i, %i)" %( ceil( self.width / self.delta ), ceil( self.height / self.delta ) )

    for o_id, frame in frame_data.items():
      oi = None
      oj = None
      ok = None
      result = frame["features"][0] * 0
      features = []
      states = []
      frames = []
      for index in xrange( len( frame["states"] ) ):
        x, y, vx, vy = frame["states"][index]
        if abs( vx * vy ) > 0.0:
          angle = momo.angle.as_angle( np.array( [vx, vy] ) )
          i, j, k = self.grid.from_world( x, y, angle )
          if i != oi or j != oj or k != ok:
            oi = i
            oj = j
            ok = k
            states.append( frame["states"][index] )
            features.append( frame["features"][index] )
            frames.append( frame["frames"][index] )
          if "feature_sum" not in frame:
            frame["feature_sum"] = frame["features"][0] * 0
          frame["feature_sum"] += frame["features"][index]
      frame["features"] = features
      frame["states"] = states
      frame["frames"] = frames
    return frame_data
