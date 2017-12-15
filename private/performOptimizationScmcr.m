function [parameters] = performOptimizationScmcr(parameters, negLogPost, iMS, J_0, options)

    [theta,J_opt,meta] = scmcr(negLogPost,parameters.MS.par0(:,iMS),options.localOptimizerOptions);


    % Assignment of results
    parameters.MS.exitflag(iMS) = meta.exitflag;
%     parameters.MS.logPost0(1, iMS) = -J_0;
    parameters.MS.logPost(iMS) = -J_opt;
    parameters.MS.par(:,iMS) = theta;
    
    parameters.MS.gradient(:,iMS) = meta.g;
    parameters.MS.hessian(:,:,iMS) = meta.H;
    
    parameters.MS.n_objfun(iMS) = meta.funEvals;
    parameters.MS.n_iter(iMS) = meta.iterations;
    
    parameters.MS.AIC(iMS) = 2*parameters.number + 2*J_opt;
    if ~isempty(options.nDatapoints)
        parameters.MS.BIC(iMS) = log(options.nDatapoints)*parameters.number + 2*J_opt;
    end
end