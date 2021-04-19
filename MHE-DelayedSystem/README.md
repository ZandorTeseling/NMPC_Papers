# Real-time MHE-base Second Order ODE

Try estimate simple second order system with mocked data and inconsistent delay.

TODO:
Create a continuous time model with known parameters to then convert to discrete, run to generate mock data.

Then use armax model with to estimate the discrete model as the time delay changes.


Absorb the delay into the model state vector.

If the delay isn't know but it has a limit n=td/dt+headroom. 
Rather than armax model where nk delay is fixed for when input starts effecting output let mhe decide and fit.  
This isn't ideal because the size of the optimisation problem has increased...   



For MHE...
x = [x(k) u(k-td) u(k_td-1) u(k_td-2) ..... u(k_td-n)]'  

Ad = [A Btd Btd-1 Btd-2 ..... Btd-n;
      0  numpy.diag(v, k=1) ]  
Bd = [0(nx+n-1,1); 1]

