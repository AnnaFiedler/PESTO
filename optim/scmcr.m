function [x,fval,meta] = scmcr(fun,x0,options)
% A regularized separable cubic trust-region method, based on
% [Cubic-regularization counterpart of a variable-norm trust-region method
% for unconstrained minimization. Martinez, Raydan. 2015]
%
% Input:
%   fun     : function pointer to a function [fval,g,H] = fun(x) where fval
%             is the function value, g the gradient, and H the hessian
%             matrix at x
%   x0      : start parameters for local search
%   options : algorithm options
%       .tol
%       .sigma0
%       .sigma_small
%       .Delta
%       .alpha
%       .rhomin
%       .rhomax
%       .rho0
%       .sigma_factor
%       .maxIter
%       .maxFunEvals
%       .barrier

% initialize values
x = x0(:);
n = size(x,1);

% initialize meta data indicating failure
meta.exitflag = -1;
fval = nan;
meta.g = nan(n,1);
meta.H = nan(n,n);
meta.iterations = 0;
meta.funEvals = 0;

% unit roundoff
epsilon = sqrt(eps);

% parameters
if nargin < 3, options = struct(); end
[tol,sigma,sigma_small,sigma_factor,Delta,alpha,rhomin,rhomax,rho,maxIter,maxFunEvals,lb,ub,barrier] = getOptions(n,options);

% check feasibility of starting point
if any(x<lb) || any(x>ub)
    return;
end

% function value and derivatives at start point
[fval,g,H] = fun(x);
jIter = 1;
jFunEvals = 1;

% check if function differentiable at start point
if isnan(fval) || isinf(fval) || any(isnan(g)) || any(isinf(g)) || any(any(isnan(H))) || any(any(isinf(H)))
    return;
end

% [Q,T] = schur(A) where A = Q*T*Q', Q unitary, T (Schur form)
% upper triangular. Can be computed e.g. via QR. For symmetric matrices,
% this is the spectral decomposition with T = D diagonal.
[Q,D] = schur(H);
b = Q'*g;
gnorm = norm(g,2);

% reserve space for solution of rotated problem
y = zeros(n,1);

% wrap function with barrier
bounded_fun = @(x,jIter) bound_fun(x,fun,lb,ub,barrier,jIter,maxIter);

% main loop
% TODO also add conditions for steplength and minimum (negative) eigenvalue
while gnorm > tol && jIter < maxIter && jFunEvals < maxFunEvals
    jIter = jIter + 1;
    
    % solve trust-region subproblem
    for j=1:n
        c0=0;c1=b(j);c2=D(j,j)/2;c3=rho(j)/6;c4=sigma/6;
        y(j) = minPoly(c0,c1,c2,c3,c4,-Delta,Delta);
    end
    
    % try step
    s = Q*y;fprintf('%d %.15f %.15f %.15f %f \n',jIter,fval,gnorm,norm(s),sigma);
    x_new = x + s;
    fval_new = bounded_fun(x_new,jIter);
    if isnan(fval_new) || isinf(fval_new) || fval_new > fval - alpha * sum(abs(y).^3)
        % no success: increase sigma
        sigma = max([sigma_small,sigma_factor*sigma]);
    else
        % success: update values
        % TODO: decrease sigma?
        sigma = sigma/(sigma_factor/1);
        
        % also compute derivatives
        % TODO: we compute fval_new twice, little overhead
        [fval_new,g_new,H_new] = fun(x_new);
        jFunEvals = jFunEvals + 1;
        
        % check validity
        if isnan(fval_new) || isinf(fval_new)...
                || any(isnan(g_new)) || any(isinf(g_new))...
                || any(any(isnan(H_new))) || any(any(isinf(H_new)))
            sigma = max([sigma_small,sigma_factor*sigma]);
        else
            % update values
            [Q_new,D_new] = schur(H_new);
            
            % update rho inspired by a third-order secant equation
            denominator = Q_new'*s;
            for j=1:n
                dj = denominator(j);
                if -epsilon < dj && dj <= 0
                    denominator(j) = -epsilon;
                elseif 0 < dj && dj < epsilon
                    denominator(j) = epsilon;
                end
            end
            rho = diag((D_new-Q_new'*H*Q_new))./denominator;
            
            % keep rho within bounds
            rho = min(max(rho,rhomin),rhomax);
            
            % update running variables
            x = x_new;
            fval = fval_new;
            g = g_new;
            H = H_new;
            Q = Q_new;
            D = D_new;
            gnorm = norm(g,2);
            
            b = Q'*g;
        end
    end
end

% meta information
if gnorm >= tol
    % did not converge
    meta.exitflag = 0;
else
    meta.exitflag = 1;
end
meta.algorithm = 'rsc';
meta.iterations = jIter;
meta.funEvals = jFunEvals;
meta.g = g;
meta.H = H;

end

%% Helper functions

function z = minPoly(c0,c1,c2,c3,c4,DeltaNeg,DeltaPos)
% compute the minimum of the function h in [DeltaNeg,DeltaPos]
h = @(z) c0 + c1*z + c2*z^2 + c3*z^3 + c4*abs(z)^3;
zs = [DeltaNeg,DeltaPos];
if c2 == 0 && c3 == 0 && c4 == 0
    z = argmin(zs,h);
elseif c2 ~= 0 && c3 == 0 && c4 == 0
    zcrt = -c1/(2*c2);
    if DeltaNeg < zcrt && zcrt < DeltaPos
        zs = [zs zcrt];
    end
    z = argmin(zs,h);
elseif c3 ~= 0 && c4 == 0
    xi = 4*c2^2-12*c3*c1;
    if xi < 0
        z = argmin(zs,h);
    else
        zcrt1 = (sqrt(xi)-2*c2)/(6*c3);
        zcrt2 = (sqrt(xi)+2*c2)/(6*c3);
        if DeltaNeg < zcrt1 && zcrt1 < DeltaPos
            zs = [zs zcrt1];
        end
        if DeltaNeg < zcrt2 && zcrt2 < DeltaPos
            zs = [zs zcrt2];
        end
        z = argmin(zs,h);
    end
elseif c4 ~= 0
    zneg = minPoly(c0,c1,c2,c3-c4,0,DeltaNeg,0);
    zpos = minPoly(c0,c1,c2,c3+c4,0,0,DeltaPos);
    z = argmin([zneg,zpos],h);
end
end

function z = argmin(zs,fun)
% value z in zs such that fun(z) is minimal among all z in zs
n = size(zs,2);
z = zs(1);
fval = fun(z);
for j=2:n
    z_new = zs(j);
    fval_new = fun(z_new);
    if fval_new < fval
        z = z_new;
        fval = fval_new;
    end
end
end

function [tol,sigma,sigma_small,sigma_factor,Delta,alpha,rhomin,rhomax,rho,maxIter,maxFunEvals,lb,ub,barrier] = getOptions(n,options)

% default values
tol = 1e-8;
sigma = 0; % start with second order problem
sigma_small = 0.1;
sigma_factor = 2;
Delta = 2;
alpha = 1*1e-4;
rhomax = 1e3*ones(n,1);
rhomin = -rhomax;
rho = 1*ones(n,1);
maxIter = 1000;
maxFunEvals = 1000;
lb = -Inf*ones(n,1);
ub = Inf*ones(n,1);
barrier = '';

% extract from input
if isfield(options,'Tol')
    tol = options.Tol;
end
if isfield(options,'Sigma0')
    sigma = options.Sigma0;
end
if isfield(options,'Sigma_small')
    sigma_small = options.Sigma_small;
end
if isfield(options,'Sigma_factor')
    sigma_factor = options.Sigma_factor;
end
if isfield(options,'Delta')
    Delta = options.Delta;
end
if isfield(options,'Alpha')
    alpha = options.Alpha;
end
if isfield(options,'Rhomax')
    rhomax = options.Rhomax;
end
if isfield(options,'Rhomin')
    rhomin = options.Rhomin;
end
if isfield(options,'Rho0')
    rho = options.Rho0;
end
if isfield(options,'MaxIter')
    maxIter = options.MaxIter;
end
if isfield(options,'MaxFunEvals')
    maxFunEvals = options.MaxFunEvals;
end
if isfield(options,'Lb')
    lb = options.Lb;
end
if isfield(options,'Ub')
    ub = options.Ub;
end
if isfield(options,'Barrier')
    barrier = options.Barrier;
end

end

function fval = bound_fun(x,fun,lb,ub,barrier,jIter,maxIter)
if ~isequal(barrier,'')
    fval = fun(x);
    fval = barrierFunction(fval, [], x, [lb, ub], jIter, maxIter, barrier);
else
    % extreme barrier
    % set fun to inf whenever conditions not fulfilled
    if any(x>ub) || any(x<lb)
        fval = inf;
    else
        fval = fun(x);
    end
end
end