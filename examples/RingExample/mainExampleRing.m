% mainExampleGauss.m shows how to use sampling methods in PESTO. The
% example problem is a smudged hyper-ring. Its properties can be altered in
% define_Gauss_LLH.m and its dimension via ringDimension.
%
% Written by Benjamin Ballnus 2/2017

% Initialize example problem
path(pathdef);
addpath(genpath([pwd filesep '..' filesep '..']));
radius = 15;
sigma = 2;
logP = @(theta) simulateRingLLH(theta,radius,sigma);
ringDimension          = 2;


% Set required sampling options for Parallel Tempering
clear opt; clear par;
par.number             = ringDimension;
par.min                = -25*ones(ringDimension,1);
par.max                = 25*ones(ringDimension,1);
par.name               = {'X_1','X_2'};

opt.obj_type           = 'log-posterior';
opt.rndSeed            = 3;
opt.nIterations        = 1e5;

% Optimization
optMS = PestoOptions();
optMS.obj_type = 'log-posterior';
optMS.objOutNumber = 1;
optMS.n_starts = 20;
optMS.comp_type = 'sequential';
optMS.mode = 'visual';
% optMS.plot_options.add_points.par = theta_true;
% optMS.plot_options.add_points.logPost = objectiveFunction(theta_true);
% optMS.plot_options.add_points.prop = nan(properties.number,1);
par = getMultiStarts(par, logP, optMS);

% Profiles
par = getParameterProfiles(par, logP, optMS);



% Using PT
opt.samplingAlgorithm     = 'PT';
opt.objOutNumber          = 1;
opt.PT.nTemps             = 3;
opt.PT.exponentT          = 4;    
opt.PT.alpha              = 0.51;
opt.PT.temperatureAlpha   = 0.51;
opt.PT.memoryLength       = 1;
opt.PT.regFactor          = 1e-4;
opt.PT.temperatureAdaptionScheme =  'Lacki15'; %'Vousden16'; %
opt.theta0                = repmat([-15*ones(ringDimension,1)],1,opt.PT.nTemps); 
opt.sigma0                = 1e5*diag(ones(1,ringDimension));

% Using DRAM
% opt.samplingAlgorithm     = 'DRAM';
% opt.objOutNumber          = 1;
% opt.DRAM.nTry             = 5;
% opt.DRAM.verbosityMode    = 'debug';    
% opt.DRAM.adaptionInterval = 1;
% opt.DRAM.regFactor        = 1e-4;
% opt.theta0                = -15*ones(ringDimension,1); 
% opt.sigma0                = 1e5*diag(ones(1,ringDimension));

% Using MALA
% opt.samplingAlgorithm     = 'MALA';
% opt.objOutNumber          = 1;
% opt.MALA.regFactor        = 1e-4;
% opt.theta0                = -15*ones(ringDimension,1); 
% opt.sigma0                = 1e5*diag(ones(1,ringDimension));

% Using PHS
% opt.samplingAlgorithm     = 'PHS';
% opt.objOutNumber          = 1;
% opt.PHS.nChains           = 3;
% opt.PHS.alpha             = 0.51;
% opt.PHS.memoryLength      = 1;
% opt.PHS.regFactor         = 1e-4;
% opt.PHS.trainingTime      = ceil(opt.nIterations / 5);
% opt.theta0                = repmat([-15*ones(ringDimension,1)],1,opt.PHS.nChains); 
% opt.sigma0                = 1e5*diag(ones(1,ringDimension));


% Perform the parameter estimation via sampling
par = rmfield(par, 'P');
par = rmfield(par, 'MS');
par = getParameterSamples(par, logP, opt);


% Visualize
figure
subplot(2,1,1); plot(squeeze(par.S.par(:,:,1))')
subplot(2,1,2); 
plotRing(); hold all
plot(squeeze(par.S.par(1,:,1))',squeeze(par.S.par(2,:,1))','.')



samplingPlottingOpt = PestoPlottingOptions();
samplingPlottingOpt.S.plot_type = 1; % Histogram
% samplingPlottingOpt.S.plot_type = 2; % Density estimate
samplingPlottingOpt.S.ind = 1; % 3 to show all temperatures
samplingPlottingOpt.S.col = [0.8,0.8,0.8;0.6,0.6,0.6;0.4,0.4,0.4];
samplingPlottingOpt.S.sp_col = samplingPlottingOpt.S.col;

plotParameterSamples(par,'1D',[],[],samplingPlottingOpt)

plotParameterSamples(par,'2D',[],[],samplingPlottingOpt)

par = getParameterConfidenceIntervals(par, [0.9,0.95,0.99]);







