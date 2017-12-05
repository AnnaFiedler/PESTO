function [x,fval,meta] = sfn(fun,x0,options)
% Saddle free Newton method, based on [Identifying and attacking the saddle
% point problem in high-dimensional non-convex optimization. Dauphin et al,
% 2014].

x = x0(:);
n = size(x,1);

tol = 1e-6;
maxIter = 1000;
maxFunEvals = 400;

gnorm = 1;

jIter = 0;
jFunEvals = 0;

while gnorm > tol && jIter < maxIter && jFunEvals < maxFunEvals
    jIter = jIter + 1;
    [fval,g,H] = fun(x);
    [Q,D] = schur(H);
    % make positive definit
    H = Q*abs(D)*Q';
    
    s = - H \ g;
    
    % find step satisfying the Wolfe conditions
    x = doWolfeStep(fun,x,s);
    
end

function x = doWolfeStep(fun,x,s)

end

