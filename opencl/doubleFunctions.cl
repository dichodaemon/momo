double t_exp( double v )
{
  double vI = v;
  double resultB = 1;
  double result = 1 + vI;
  int i = 2;
  while( resultB - result != 0 )
  {
    vI   = vI * v / i;
    resultB = result;
    result  += vI;
    i++;
  }
  return result;
}
