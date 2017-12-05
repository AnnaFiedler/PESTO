function [x,fval,meta] = rsc(fun,x0,options)
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
meta.exitflag = -1;
meta.g = zeros(n,1);
meta.H = zeros(n,n);
meta.iterations = 0;
meta.funEvals = 0;
% parameters
if nargin < 3, options = struct(); end
[tol,sigma,sigma_small,sigma_factor,Delta,alpha,rhomin,rhomax,rho,maxIter,maxFunEvals,lb,ub,barrier] = getOptions(n,options);
% check feasibility of starting point
if any(x<lb) || any(x>ub)
    return;
end
[fval,g,H] = fun(x);
% check if function differentiable at starting point
if isnan(fval) || isinf(fval) || any(isnan(g)) || any(isinf(g)) || any(any(isnan(H))) || any(any(isinf(H)))
    return;
end
% [U,T] = schur(A) where A = U*T*U', U unitary, T (Schur form)
% upper triangular. Can be computed e.g. via QR.
[Q,D] = schur(H);
b = Q'*g;
gnorm = norm(g,2);

% reserve space for solution of rotated problem
y = zeros(n,1);

% wrap function with barrier
bounded_fun = @(x,jIter) bound_fun(x,fun,lb,ub,barrier,jIter,maxIter);

jIter = 0;
jFunEvals = 0;

while gnorm > tol && jIter < maxIter && jFunEvals < maxFunEvals
    jIter = jIter + 1;
    
    % solve trust-region subproblem
    for j=1:n
        c0=0;c1=b(j);c2=D(j,j)/2;c3=rho(j)/6;c4=sigma/6;
        y(j) = minPoly(c0,c1,c2,c3,c4,-Delta,Delta);
    end
    
    % try step
    s = Q*y;fprintf('%d %.15f %.15f %.15f \n',jIter,fval,gnorm,norm(s));
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
            
            % update rho inspired by secant equation
            rho = diag((D_new-Q_new'*H*Q_new))./(Q_new'*s);
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
    if zcrt > DeltaNeg && zcrt < DeltaPos
        zs = [zs zcrt];
    end
    z = argmin(zs,h);
elseif c3 ~= 0 && c4 == 0
    xi = 4*c2^2-12*c3*c1;
    if xi < 0
        z = argmin(zs,h);
    else
        zlmin = argmin([(sqrt(xi)-2*c2)/(6*c3),(sqrt(xi)+2*c2)/(6*c3)],h);
        if zlmin > DeltaNeg && zlmin < DeltaPos
            zs = [zs zlmin];
        end
        z = argmin(zs,h);
    end
elseif c4 ~= 0
    zpos = minPoly(c0,c1,c2,c3+c4,0,0,DeltaPos);
    zneg = minPoly(c0,c1,c2,c3-c4,0,DeltaNeg,0);
    z = argmin([zpos,zneg],h);
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
Delta = 5;
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