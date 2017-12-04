function [x,fval,meta] = rsc(fun,x0)
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

% initialize values
x = x0(:);
n = length(x);
[fval,g,H] = fun(x);
[Q,D] = schur(H);
b = Q'*g;

% reserve space for solution of rotated problem
y = zeros(n,1);

% parameters
tol = 1e-8;
sigma = 0; % start with second order problem
sigma_small = 0.1;
Delta = 2;
alpha = 0.1;
rhomax = 1e3*ones(n,1);
rhomin = -rhomax;
rho = 1*ones(n,1);
sigma_factor = 10;

while norm(g) > tol
    
    % step 2
    % solve subproblem
    for j=1:n
        c0=0;c1=b(j);c2=D(j,j)/2;c3=rho(j)/6;c4=sigma/6;
        z = minPoly(c0,c1,c2,c3,c4,-Delta,Delta);
        y(j) = z;
    end
    
    % step 3
    s = Q*y;
    x_new = x + s;
    fval_trial = fun(x_new);
    if fval_trial > fval - alpha * sum(abs(y).^3)
        % no success: increase sigma
        sigma = max([sigma_small,sigma_factor*sigma]);
    else
        % success: update values
        % also compute derivatives
        [fval_new,g_new,H_new] = fun(x);
        % update values
        [Q_new,D_new] = schur(H_new);
        
        % update rho inspired by secant equation
        rho = diag((D_new-Q_new'*H*Q_new))./(Q_new'*s);
        rho = min(max(rho,rhomin),rhomax);
        
        % update running variables
        x = x_new;
        fval = fval_new;
        g = g_new;
        H = H_new;
        Q = Q_new;
        D = D_new;
        
        b = Q'*g;
    end
end

% [U,T] = schur(M) where A = U*T*U', U unitary, T (Schur form)
% upper triangular. Can be computed e.g. via QR.


% meta information
meta.exitflag = 1;
meta.algorithm = 'rsc';

end

function z = minPoly(c0,c1,c2,c3,c4,DeltaNeg,DeltaPos)
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
        c3_new = c3 + c4;
        DeltaNeg_new = 0;
        DeltaPos_new = DeltaPos;
        zpos = minPoly(c0,c1,c2,c3_new,0,DeltaNeg_new,DeltaPos_new);
        c3_new = c3 - c4;
        DeltaNeg_new = DeltaNeg;
        DeltaPos_new = 0;
        zneg = minPoly(c0,c1,c2,c3_new,0,DeltaNeg_new,DeltaPos_new);
        z = argmin([zpos,zneg],h);
    end
end

function z = argmin(zs,fun)
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

