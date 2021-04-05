# Real-time MHE-base for a sin wave

Try and estimate sin wave disturbance into body, for feedforward and corrections for MPC.

Replay 100Hz logged data and predict roughly 400ms.


% Initialize guesses  
p = [];  
p(0) = mean(y);         % vertical shift  
p(1) = maxAmplitude;    % amplitude estimate  
p(2) = maxFrequency;    % phase estimate  
p(3) = 0;               % phase shift (no guess)  
p(4) = 0;               % trend (no guess)  

% Create model  
f = @(p) p(1) + p(2)*sin( p(3)*2*pi*t+p(4) ) + p(5)*t;  
ferror = @(p) sum((f(p) - y).^2);  
