import time


times   = {}
count   = {}
last    = {}
parent  = {}
son     = {}
current = []

def clear():
  global times, last, parent, current
  times   = {}
  count   = {}
  last    = {}
  parent  = {}
  son     = {}
  current = []

def tick( name ):
  last[name] = time.time()
  if not name in times:
    times[name] = 0
    count[name] = 0
    son[name] = []
  if len( current ) > 0:
    if not name in parent:
      parent[name] = current[-1]
      son[current[-1]].append( name )
    elif parent[name] != current[-1]:
      raise "Problems here"
  current.append( name )

def tack( name ):
  t = time.time() - last[name]
  times[name] += t
  count[name] += 1
  if name != current.pop():
    raise "Problems here"

def stats( name, level = 0 ):
  indent = "\t" * level
  tmp = "%s%s - Time(total): %f Time(average): %f Frequency: %f" % (indent, name, times[name], times[name] / count[name], count[name] / times[name] )
  result = [tmp]
  for s in son[name]:
    result.extend( stats( s, level + 1 ) )
  return result
