import numpy as np
import numpy.random as rng
import theano
import theano.tensor as T

'''
UORO implementation on toy dataset.  

ds[t]/d(theta) is (M x M^2).  

This rank-1 factorizes to: 

(M x 1) (1 x M^2)



-Takes x[t+1], s[t], last_params.  

-Takes s'[t], s[t].  

-Do backprop on loss with respect to s[t] and last_params.  

state: 32x512.  
x: 32x1





Two types of state going forward: 

state, tilda_state.  

The tilda_state keeps getting random directions injected on each step and normalized to try to keep them similarly scaled.  


What does it mean to do backprop with respect to v?  

=======

-Forward function is F(v) where v includes parameters and x.  

-Tangent forward propagation is like (F(v + eps*dv) - F(v)) / eps.  Computes how much F changes along the direction dv.  

-Each step of forward produces new state s[t+1] and new output o[t+1].  


=========

s_tilda is (m x 1)
theta_tilda is (1 x m^2)

-The tilda things are the online forward estimate of the gradient ds[t]/dtheta.  

-s[t] are the states going forward.  

-Do s[t] and s_tilda cross paths?  

-s_tilda seems to depend on s, but s doesn't depend on s_tilda.  

-s_tilda evolves based on v, old s_tilda, and old s.  

-Mystery quantities: 
    -delta_theta: local gradient of loss wrt params.  
    -delta_s: local gradient of loss wrt states.  
    -v
    -delta_theta_g
    
-gradient estimate depends on delta_s, s_tilda, theta_tilda, delta_theta.  

-So I think dL/ds is delta_s and dL/dtheta is delta_theta.  So these are like local gradient estimates wrt the local loss.  

(1xm)*(1xm) (m x m) + (m x m)

(local_s_grad * low_rank_s) * (low_rank_theta) + theta_local

-Adding in theta local makes perfect sense.  

(1 x m)(m x m) --> (m x m).  

-Step right before gradient is confusing.  

dL/do * do/ds * ds/dtheta
(1xm)(mxm)(mxm^2)
(1xm)(mxm)(mx1)(1 x m^2)

dL/ds is (1xm)

(1xm)(mx1)((1 x m^2)

dL/do is delta_s

Key intuition is that you want to compute dL/dW.  You can compute this as:

dL/ds * ds/dW

but compute out the left side first, corresponding to a low-rank trick.  

(1xm)(mx1)(1 x m^2)

-Does s_tilda have one value across minibatch, or separate value for each example?  

-delta_theta_g is like the update to the low-rank approximation part for (1 x m^2)

-The update to theta_tilda is v' * dstate_out / dtheta.  

-theta_tilda is (m x m).  v is (m x 1).  dstate_out / dtheta is (m x m^2).  

(1 x m)(m x m^2) --> (1 x m^2).  

(v) * (ds / dtheta)

-Is this the same as computing dot(v,s) which is a scalar, and then computing gradient wrt theta.  

(1 x m)(m x 1)(1 x m^2)



'''

theta_trans = theano.shared(rng.normal(size=(512+1,512)).astype('float32'))
theta_em = theano.shared(rng.normal(size=(512,1)).astype('float32'))

def join2(a,b):
    return T.concatenate([a,b], axis=1)

def uoro_step(x, target, state, state_tilda, theta_tilda_em, theta_tilda_trans):

    state_next = T.dot(join2(state,x), theta_trans)
    
    target_est = T.dot(state, theta_em)

    loss = T.sum(T.sqr(target_est - target))

    theta_delta_trans = T.grad(loss, theta_trans)
    theta_delta_em = T.grad(loss, theta_em)
    state_delta = T.grad(loss, state)

    
    new_gradient_trans = T.dot(T.dot(state_delta,state_tilda.T), theta_tilda_trans) + theta_delta_trans

    new_gradient_em = T.dot(T.dot(state_delta,state_tilda.T), theta_tilda_em) + theta_delta_em

    v = srng.normal(size=state_delta.shape)




