all: learn test

#test : test_t_3_1 test_t_3_1000

#learn: learn_t_3_1 learn_t_3_1000

#test_t_3_1: 
	#bin/test crossing irl abbeel 3 1 --ids 100,101,102,103 --delta 0.25 --velocity 0.04 --num_random 5 --frame_skip 3 --start_frame 100
#test_t_3_1000:
	#bin/test crossing irl abbeel 3 1000 --ids 100,101,102,103 --delta 0.25 --velocity 0.04 --num_random 5 --frame_skip 3 --start_frame 100
#learn_t_3_1 :
	#bin/learn crossing irl abbeel 3 1 --ids 90,91,92,93 --delta 0.25
#learn_t_3_1000:
	#bin/learn crossing irl abbeel 3 1000 --ids 90,91,92,93 --delta 0.25

test:
	bin/test long_sparse irl abbeel 3 20 --ids 30 --delta 0.25 --velocity 0.024 --num_random 1 --frame_skip 3 --start_frame 100

learn:
	bin/learn long_sparse irl abbeel 3 20 --ids 61,1,53 --delta 0.25
	#bin/learn long_sparse irl stork 3 10 --ids 30,61,1,53 --delta 0.25
	#bin/learn long_sparse irl stork 3 20 --ids 30,61,1,53 --delta 0.25
	#bin/learn long_sparse irl stork 3 30 --ids 30,61,1,53 --delta 0.25
	#bin/learn long_sparse irl stork 3 40 --ids 30,61,1,53 --delta 0.25
	#bin/learn long_sparse irl stork 3 1000 --ids 30,61,1,53 --delta 0.25
	#bin/learn long irl stork 3 10 --ids 427,296,170,75 --delta 0.25
	#bin/learn long irl stork 3 20 --ids 427,296,170,75 --delta 0.25
	#bin/learn long irl stork 3 30 --ids 427,296,170,75 --delta 0.25
	#bin/learn long irl stork 3 40 --ids 427,296,170,75 --delta 0.25
	#bin/learn long irl stork 3 1000 --ids 427,296,170,75 --delta 0.25
