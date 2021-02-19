function [ cfg ] = mvpalab_init()
%INITIALIZE Summary of this function goes here
%   Detailed explanation goes here
clear
clc
cfg = [];

fprintf('<strong> > Initializing MVPAlab toolbox: </strong>\n');

%% Datafiles and paths:
cfg.study.dataPaths = {{},{};{},{}};
cfg.study.dataFiles = {{},{};{},{}};
cfg.study.conditionIdentifier = {'','';'',''};
cfg.analysis = 'MVPA';
cfg.study.mvccDirection = 'both';

%% Configuration - Feature extraction:

% Power envelope as features:
cfg.feature         = 'voltage';
cfg.powenv.method   = 'analytic';
cfg.powenv.uplow    = 'upper';
cfg.powenv.length   = 5;

% Supertrial generation:
cfg.trialaver.flag     = false;
cfg.trialaver.ntrials  = 0;
cfg.trialaver.order    = 'rand';

% Match class size:
cfg.classsize.match = false;
cfg.classsize.matchkfold = false;

%% Configuration - Feature selection:

% Feature selection methods:
% 'pca' - Principal Component Analysis
% 'pls' - Partial Least Squares
% 'lda' - Linear Discriminant Analysis (in progress)
cfg.dimred.method = 'none';
cfg.dimred.flag   = false;
cfg.dimred.ncomp  = 0;

%% Configuration - Data normalization:

% Notmalization methods:
% 0 - raw data
% 1 - z-score (across features)
% 2 - z-score (across time)
% 3 - z-score (across trials)
% 4 - std_nor (across trials)

cfg.normdata = 0; 

%% Configuration - Data smoothing:
cfg.smoothdata.method   = 'none';
cfg.smoothdata.window   = 1;

%% Configuration - Analysis timing:
cfg.tm.tpstart   = 0;
cfg.tm.tpend     = 0;
cfg.tm.tpsteps   = 1;

%% Configuration - Classification procedure:

% Classification algorithms:
% 'svm' - Support Vector Machine.
% 'lda' - Linear Discriminant Analysis.
cfg.classmodel.method = 'svm';
cfg.classmodel.kernel = 'linear';

% Analysis configuration:
cfg.classmodel.tempgen = false;
cfg.classmodel.parcomp = false;
cfg.classmodel.permlab = false;

% Performance metrics:
cfg.classmodel.roc       = false;
cfg.classmodel.auc       = false;
cfg.classmodel.confmat   = false;
cfg.classmodel.precision = false;
cfg.classmodel.recall    = false;
cfg.classmodel.f1score   = false;
cfg.classmodel.wvector   = false;
    

% Optimization configuration:
cfg.classmodel.optimize.flag = false;
cfg.classmodel.optimize.params = {'BoxConstraint'};
cfg.classmodel.optimize.opt = struct('Optimizer','gridsearch',...
    'ShowPlots',false,...
    'Verbose',0,...
    'Kfold', 5);

%% Configuration - Cross-validation procedure:

% Classification algorithms:
% 'kfold' - K-Fold cross-validation.
% 'loo'   - Leave-one-out cross-validation.

cfg.cv.method  = 'kfold';
cfg.cv.nfolds  = 5;

%% Configuration - Statistic for permutation method:
cfg.stats.flag   = 0;
cfg.stats.nper   = 100;
cfg.stats.nperg  = 1e5;
cfg.stats.pgroup = 99.9;
cfg.stats.pclust = 99.9;
cfg.stats.shownulldis = 0;

%% Configure sliding filter analysis:
% Flag:
cfg.sf.flag = 0;
cfg.sf.filesLocation = '';

% Frequency limits:
cfg.sf.lfreq = 0;              % Analysis inferior limit (Hz).
cfg.sf.hfreq = 40;             % Analysis superior limit (Hz).

% Filter design:
cfg.sf.ftype = 'bandstop';      % Filter type.
cfg.sf.wtype = 'blackman';      % Window type.
cfg.sf.bw    = 2;               % Filter bandwidth (Hz).
cfg.sf.hbw   = cfg.sf.bw/2;     % Halfband width (Hz).
cfg.sf.order = 1408;            % Filter order.

% Frequency steps:
cfg.sf.fspac = 'log';           % Linear or logarithmic (lin/log.)
cfg.sf.nfreq = 40;              % Number of steps - log (Hz).

% For data import (mvpalab_load):
if ~cfg.sf.flag; cfg.sf.nfreq = 1; end


%% Other configuration - Parallel computing:
fprintf('<strong> > Checking parallel computing: </strong>');
% if license('test','Distrib_Computing_Toolbox')
%     fprintf('- Toolbox installed.\n');
%     cfg.classmodel.parcomp = true;
%     p = gcp('nocreate'); 
%     if isempty(p)
%         parpool;
%     end
% else
%     disp('- Toolbox not available');
%     cfg.classmodel.parcomp = false;
% end

fprintf('<strong> > MVPAlab is ready! </strong>\n');

%% PCA rank warning disabled. 
warning('off','stats:pca:ColRankDefX');

%% Software version:
cfg.version = 'BETA v0.1';

end

