function cfg = mvpalab_preparesf(cfg)

cfg.classmodel.tempgen = false; % Disable temporal generalization.
cfg.sf.flag = true; % Enable SF just in case.

cfg.sf.savefolder = [cfg.location filesep 'results' filesep 'diffMaps' filesep];
cfg = mvpalab_genfreqvec(cfg);  % Generate cutoff frequencies.
cfg = mvpalab_sfmetrics(cfg);   % Update performace metrics.

% If the filesLocation folder exists, remove it for the new analysis. This
% is required in order to prenvent errors if the user repeat the analysis
% with less frequency steps.

if exist(cfg.sf.filesLocation,'dir')
    rmdir(cfg.sf.filesLocation,'s');
end

mvpalab_mkdir(cfg.sf.filesLocation); % Create data folder.
mvpalab_mkdir(cfg.sf.savefolder);    % Create result folder.
mvpalab_mkdir([cfg.sf.savefolder 'other' filesep]);

end

