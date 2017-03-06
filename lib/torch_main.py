'''
Initially don't use x, just use previous state.  

s[t] = W*s[t-1]
y[t] = sum(W*s[t])

assume that: 

Given variables: 
    observed_target: Nx1
    last_state: NxM
    theta: (MxM)

Intermediate variables:
    next_state: NxM
    estimated_target: Nx1
    delta_state: NxM
    delta_theta: MxM
    new_gradient: MxM
    v: Mx1
    next_state_tilde: Mx1
    delta_theta_g: MxM
    theta_tilda: MxM

'''

import torch
from torch.autograd import Variable

N = 64
M = 16

observed_target = Variable(torch.ones((N,1)), requires_grad=True)
last_state = Variable(torch.ones((N,M)), requires_grad=True)
theta = Variable(torch.ones((M,M)), requires_grad=True)
state_tilde = Variable(torch.ones((M,1)), requires_grad=True)
theta_tilde = Variable(torch.ones((1,M,M)), requires_grad=True)
epsilon = 0.0001

next_state = torch.mm(last_state, theta)

estimated_target = torch.mm(next_state, theta).sum(1)

loss = ((estimated_target - observed_target)**2).sum()

loss.backward()

delta_state = last_state.grad
delta_theta = theta.grad


#(N x M)(M x 1).  
#(N x 1)(1 x M^2) --> (N x M^2)
new_gradient_1 = torch.mm(delta_state, state_tilde)
new_gradient_2 = torch.mm(new_gradient_1, theta_tilde.view(1,M**2)).sum(0).view(M,M)

new_gradient = new_gradient_2 + delta_theta

v = Variable(torch.rand(M,1))

#THIS STEP IS DEFINITELY WRONG.  NOT SURE WHAT FORWARD DIFF IS SUPPOSED TO BE HERE.  
next_state_tilde = (last_state - state_tilde.expand(M,N).transpose(0,1)).sum(0).transpose(0,1)

#state is (NxM).  v is (Mx1).  Need to get a scalar.  

#THIS STEP IS ALSO DEFINITELY WRONG.  
vs_dot_loss = 10.0 * torch.mm(last_state, v).sum()

#THIS STEP IS WRONG BUT FOR TORCH REASONS.  HAVENT'T FIGURED OUT HOW TO TAKE GRAD WRT DIFFERENT LOSSES YET
vs_dot_loss.backward()

delta_theta_g = theta.grad

rho_0 = torch.sqrt(torch.norm(theta_tilde) / (torch.norm(next_state_tilde)+epsilon)) + epsilon

rho_1 = torch.sqrt(torch.norm(delta_theta_g) / (torch.norm(v) + epsilon)) + epsilon


rho_0_a = rho_0.expand(16).contiguous().view(16,1)
rho_1_a = rho_1.expand(16).contiguous().view(16,1)

next_state_tilde = rho_0_a.expand(M,1)*next_state_tilde + rho_1_a.expand(M,1)*v

rho_0_b = rho_0.expand(M).contiguous().view(M,1).expand(M,M)
rho_1_b = rho_1.expand(M).contiguous().view(M,1).expand(M,M)


next_theta_tilde = theta_tilde/rho_0_b + delta_theta_g/rho_1_b






