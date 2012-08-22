import numpy as np

good = [  
  [9.77020647e+00, -2.77595992e+00],  
  [9.53867992e+00, -2.80005176e+00],  
  [9.29900412e+00, -2.80022155e+00],  
  [9.06966333e+00, -2.78844240e+00],  
  [8.83566558e+00, -2.79934540e+00],  
  [8.59990418e+00, -2.78076090e+00],  
  [8.36633278e+00, -2.78255420e+00],  
  [8.12993359e+00, -2.77172321e+00],  
  [7.89933492e+00, -2.77105083e+00],  
  [7.66392420e+00, -2.75760676e+00],  
  [7.43265435e+00, -2.74295328e+00],  
  [7.19527590e+00, -2.76981334e+00],  
  [6.96178705e+00, -2.78174380e+00],  
  [6.73039924e+00, -2.80601472e+00],  
  [6.49771485e+00, -2.81949252e+00],  
  [6.26773784e+00, -2.81860774e+00],  
  [6.03053856e+00, -2.80029022e+00],  
  [5.79920001e+00, -2.81394760e+00],  
  [5.56634522e+00, -2.83602526e+00],  
  [5.34195644e+00, -2.80990262e+00],  
  [5.10165136e+00, -2.80661294e+00],  
  [4.87161841e+00, -2.78545780e+00],  
  [4.64300773e+00, -2.75378485e+00],  
  [4.40552044e+00, -2.74539305e+00],  
  [4.17404654e+00, -2.74691643e+00],  
  [3.94112784e+00, -2.73342374e+00],  
  [3.70825737e+00, -2.74733600e+00],  
  [3.47497278e+00, -2.73582479e+00],  
  [3.24345352e+00, -2.70476874e+00]  
]

bad = [
  [7.6195723e+01,   2.5341403e+00], 
  [7.5385380e+01,   2.4498188e+00], 
  [7.4546514e+01,   2.4492246e+00], 
  [7.3743822e+01,   2.4904516e+00], 
  [7.2924830e+01,   2.4522911e+00], 
  [7.2099665e+01,   2.5173368e+00], 
  [7.1282165e+01,   2.5110603e+00], 
  [7.0454768e+01,   2.5489688e+00], 
  [6.9647672e+01,   2.5513221e+00], 
  [6.8823735e+01,   2.5983763e+00], 
  [6.8014290e+01,   2.6496635e+00], 
  [6.7183466e+01,   2.5556533e+00], 
  [6.6366255e+01,   2.5138967e+00], 
  [6.5556397e+01,   2.4289485e+00], 
  [6.4742002e+01,   2.3817762e+00], 
  [6.3937082e+01,   2.3848729e+00], 
  [6.3106885e+01,   2.4489842e+00], 
  [6.2297200e+01,   2.4011834e+00], 
  [6.1482208e+01,   2.3239116e+00], 
  [6.0696848e+01,   2.4153408e+00], 
  [5.9855780e+01,   2.4268547e+00], 
  [5.9050664e+01,   2.5008977e+00], 
  [5.8250527e+01,   2.6117530e+00], 
  [5.7419322e+01,   2.6411243e+00], 
  [5.6609163e+01,   2.6357925e+00], 
  [5.5793947e+01,   2.6830169e+00], 
  [5.4978901e+01,   2.6343240e+00], 
  [5.4162405e+01,   2.6746132e+00], 
  [5.3352087e+01,   2.7833094e+00] 
]


scale  = 3.5
offset = np.array( [12., 3.5] )

for i in xrange( len( good ) ):
  g = np.array( good[i] )
  b = np.array( bad[i] )
  guess = b / scale - offset
  print g, guess


#for i in xrange( 1, len( good ) ):
  #g1 = np.array( good[i - 1] )
  #g2 = np.array( good[i] )
  #b1 = np.array( bad[i - 1] )
  #b2 = np.array( bad[i] )

  #gd = g2 - g1
  #bd = b2 - b1

  #print bd[0] / gd[0], bd[1] / gd[1]
 
start = np.array( [80., 15.] )
goal  = np.array( [20., 25.] )

print "*" * 80
print start / scale - offset, goal / scale - offset