function [cfg,diffMap,stats] = mvpalab_sfilter(cfg)
%% Initialize analysis:

savefolder = [cfg.location filesep 'results' filesep 'diffMaps' filesep];
cfg.classmodel.tempgen = false; % Disable temporal generalization.
cfg = mvpalab_genfreqvec(cfg);  % Generate cutoff frequencies.
cfg = mvpalab_sfmetrics(cfg);   % Update performace metrics.

mvpalab_mkdir(cfg.sf.filesLocation); % Create data folder.
mvpalab_mkdir(savefolder);           % Create result folder.
mvpalab_mkdir([savefolder 'sfilter' filesep]);

%% Sliding filter analysis:

% Import, prepare and filter datasets:
cfg = mvpalab_import(cfg);

% Compute analysis for each frequency band:
if strcmp(cfg.analysis,'MVPA')
    [performance_maps,cfg] = mvpalab_mvpa(cfg);
elseif strcmp(cfg.analysis,'MVCC')
    [performance_maps,cfg] = mvpalab_mvcc(cfg);
end

save([savefolder 'sfilter' filesep 'performance_maps.mat'],...
    'performance_maps','-v7.3');

% Generate permuted maps for each frequency band if needed:
if cfg.stats.flag
    
    if strcmp(cfg.analysis,'MVPA')
        [permuted_maps,cfg] = mvpalab_permaps(cfg);
    elseif strcmp(cfg.analysis,'MVCC')
        [permuted_maps,cfg] = mvpalab_cpermaps(cfg);
    end
    
    save([savefolder 'sfilter' filesep 'permuted_maps.mat'],...
        'permuted_maps','-v7.3');
    
end

%% MVPA analysis:

% Time-resolved MVPA:
cfg.sf.flag = false;
[cfg,~,fv] = mvpalab_import(cfg);

if strcmp(cfg.analysis,'MVPA')
    [performance,cfg] = mvpalab_mvpa(cfg,fv);
elseif strcmp(cfg.analysis,'MVCC')
    [performance,cfg] = mvpalab_mvcc(cfg,fv);
end

save([savefolder 'sfilter' filesep 'performance.mat'],...
    'performance','-v7.3');

% Chance level:
cfg.classmodel.permlab = true;

if strcmp(cfg.analysis,'MVPA')
    [permuted_performance,cfg] = mvpalab_mvpa(cfg,fv);
elseif strcmp(cfg.analysis,'MVCC')
    [permuted_performance,cfg] = mvpalab_mvcc(cfg,fv);
end

cfg.sf.flag = true;
cfg.classmodel.permlab = false;
save([savefolder 'sfilter' filesep 'permuted_performance.mat'],...
    'permuted_performance','-v7.3');

%% Sliding filter analysis - Generate diffMaps:

[diffMap,perdiffMap,cfg] = mvpalab_gendiffmap(...
    cfg,performance.(cfg.sf.metric),...
    performance_maps.(cfg.sf.metric),...
    permuted_performance.(cfg.sf.metric),...
    permuted_maps.(cfg.sf.metric));

result = diffMap.(cfg.sf.metric);
save([savefolder 'result.mat'],'result','cfg','-v7.3');

% Remove time_resolved generated by MVPA analysis:
s = rmdir([cfg.location filesep 'results' filesep 'time_resolved'],'s');

%% Compute permutation test if needed Permutation test:
if cfg.stats.flag
    save([savefolder 'sfilter' filesep 'perdiffMap.mat'],...
        'perdiffMap','-v7.3');
    stats = mvpalab_permtest(cfg,diffMap,perdiffMap);
    save([savefolder 'stats.mat'],'stats','-v7.3');
end
end

