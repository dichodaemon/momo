load( "pedstreams.mat" )

data  = [];
count = 1;
for i = 1:size( obsdata, 2 )
  for j = 1:size( obsdata( i ).z, 2 )
    if obsdata( i ).zvalid( j ) == 1
      v = obsdata( i ).z( :, j );
      data( :, count ) = [int32( j ), j * 0.5, int32( i ), v( 1 ), v( 2 )];
      count = count + 1;
    end
  end
end

data = data';

save( "output.txt", "data", "-ascii" )
