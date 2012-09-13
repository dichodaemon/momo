all: learn test

test : test_t_3_1 test_t_3_1000

learn: learn_t_3_1 learn_t_3_1000

test_t_3_1: 
	bin/test crossing irl abbeel 3 1 --ids 100,101,102,103 --delta 0.25 --velocity 0.04 --num_random 5 --frame_skip 3 --start_frame 100
test_t_3_1000:
	bin/test crossing irl abbeel 3 1000 --ids 100,101,102,103 --delta 0.25 --velocity 0.04 --num_random 5 --frame_skip 3 --start_frame 100
learn_t_3_1 :
	bin/learn crossing irl abbeel 3 1 --ids 90,91,92,93 --delta 0.25
learn_t_3_1000:
	bin/learn crossing irl abbeel 3 1000 --ids 90,91,92,93 --delta 0.25
