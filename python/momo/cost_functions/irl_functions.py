import momo
import numpy as np

def preprocess_data( module, data, o_ids ):
  first = min( o_ids )
  last  = max( o_ids )
  min_x = 0.
  min_y = 0.
  max_x = 0.
  max_y = 0.
  frame_data = {}
  process = False
  last_processed = 1000000
  for frame in momo.frames( data ):
    tmp = []
    for o_frame, o_time, o_id, o_x, o_y, o_dx, o_dy in frame:
      if o_id in o_ids:
        process = True
      if o_id == last:
        last_processed = o_frame
      if process:
        if o_x < min_x:
          min_x = o_x
        if o_y < min_y:
          min_y = o_y
        if o_x > max_x:
          max_x = o_x
        if o_y > max_y:
          max_y = o_y
        tmp.append( [o_id, np.array( [o_x, o_y, o_dx, o_dy] )] )
    if process:
      for i in xrange( len( frame ) ):
        o_id, o = tmp.pop( 0 )
        feature = module.compute( o, o, [t[1] for t in tmp] )
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
    if o_frame - last_processed > 10:
      break
  width = ( max_x - min_x ) * 1.01
  height = ( max_y - min_y ) * 1.01
  if width > height:
    delta  = width / 64
  else:
    delta = height / 64

  grid = momo.planning.grid( 
    min_x, min_y, width, height, delta
  )

  width = grid.width
  height = grid.height

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
        i, j, k = grid.from_world( x, y, angle )
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
  return min_x, min_y, width, height, delta, grid, frame_data

