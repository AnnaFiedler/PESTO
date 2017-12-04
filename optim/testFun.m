function [varargout] = testFun(x,index)
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
    varargout{2} = 2*x;
    varargout{3} = 2*eye(length(x));
end

