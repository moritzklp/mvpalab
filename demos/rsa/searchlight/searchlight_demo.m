%% MVPAlab TOOLBOX - (searchlight_demo.m)
% -------------------------------------------------------------------------
% Brain, Mind and Behavioral Research Center - University of Granada.
% Contact: dlopez@ugr.es (David Lopez-Garcia)
% -------------------------------------------------------------------------

%% Initialize project and run configuration file:

cfg = mvpalab_init();
run cfg_file;

%% Load mask and data:

[cfg,data,mask] = mvpalab_import_fmri(cfg);

%% Compute searchlight analysis:

[result,stats,cfg] = mvpalab_searchlight(cfg,mask,data);

%% Plot the results:

% run rsa_plot;

%% Save cfg file:

% mvpalab_savecfg(cfg);