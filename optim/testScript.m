clear

x.p=1*ones(17,1);
nparams = struct('maxit',1000,'toler',1.0e-4,'method','direct');
fun = @testFun;

addpath('../dfo_tests');
% fun = @TF.f_rosenbrock;



[exdir,~,~]=fileparts(which('mainJakstatSignaling.m'));
addpath(genpath('../examples'));
%% Data
% Experimental data is read out from an .xls-file and written to an AMICI
% object which is used for the ODE integration
datatable         = xlsread(fullfile(exdir,'pnas_data_original.xls'));
amiData.t         = datatable(:,1);       % time points
amiData.Y         = datatable(:,[2,4,6]); % measurement
amiData.condition = [1.4,0.45];           % initial conditions
amiData.Sigma_Y   = NaN(size(amiData.Y)); % preallocation of variances
amiData           = amidata(amiData);     % calling the AMICI routine

% objective function
fun = @(theta) logLikelihoodJakstat(theta, amiData);
% x.p = ones(17,1);
% 
% 
% tic
% [inform,x] = Newton(fun,x,nparams);
% toc
% x.f
% 
tic
[p,fval,meta] = rsc(fun,x.p);
toc
fval,meta.iterations
%
% options=optimoptions('fmincon','Algorithm','trust-region-reflective','HessianFcn','objective','SpecifyObjectiveGradient',true,'Display','iter');
% tic
% [p,fval,exitflag,meta] = fmincon(fun,x.p,[],[],[],[],[],[],[],options);
% toc
% fval