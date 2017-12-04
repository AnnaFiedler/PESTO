function [varargout] = testFun(x)
if nargin > 1
    if index==1
        varargout{1} = sum(x.^2);
    elseif index==2
        varargout{1} = 2*x;
    else
        varargout{1} = sparse(2*eye(length(x)));
    end
else
    varargout{1} = sum(x.^2);
    if nargout > 1
        varargout{2} = 2*x;
        if nargout > 2
            varargout{3} = 2*eye(length(x));
        end
    end
end
% 
% syms f(x,y)
% f(x,y) = x^4/4+y^4/4 - 5*x^3/3 - 5*y^3/3;
% 
% g(x,y) = jacobian(f,[x,y]);
% H(x,y) = hessian(f,[x,y]);
% 
% f = matlabFunction(f);
% g = matlabFunction(g);
% H = matlabFunction(H);
% 
% varargout{1} = f(p(1),p(2));
% varargout{2} = g(p(1),p(2))';
% varargout{3} = H(p(1),p(2));