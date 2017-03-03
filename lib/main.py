'''
UORO implementation on toy dataset.  

ds[t]/d(theta) is (M x M^2).  

This rank-1 factorizes to: 

(M x 1) (1 x M^2)



-Takes x[t+1], s[t], last_params.  

-Takes s'[t], s[t].  

-Do backprop on loss with respect to s[t] and last_params.  



'''

def step(x, state, W):



    return est_target, next_state

def uoro_step(x, target, last_state, W, state_diff, W_diff):



