%% Save cfg state:
save([cfg.location '/cfg.mat'],'cfg');

%% Update cutoff frequency vector:
cfg = mvpalab_genfreqvec(cfg);

%% Create result and stats folder
mvpalab_mkdir([cfg.location '/results']);
mvpalab_mkdir([cfg.location '/stats']);
mvpalab_mkdir(cfg.sf.filesLocation);

%% Load data, generate conditions and feature extraction:
cfg = mvpalab_import(cfg);

%% Sliding filter analysis for each frequency band - MVCC:
cfg.sf.flag = true;
[accmap,cfg] = mvpalab_mvpa(cfg);

%% Save accmaps and cfg structure:
save([cfg.location filesep 'results' filesep ...
    'accmap.mat'],'accmap','-v7.3');
save([cfg.location filesep 'cfg.mat'],'cfg','-v7.3');

%% Generate permuted maps for each frequency band:
if cfg.stats.flag
    [permaps,cfg] = mvpalab_permaps(cfg);
    if cfg.stats.savepmaps
        save([cfg.location filesep 'stats' filesep...
            'permaps.mat'],'permaps','-v7.3');
    end
end

%% Load data, generate conditions and feature extraction:
cfg.sf.flag = false;
[cfg,data,fv] = mvpalab_import(cfg);

%% MVCC analysis:
[acc,cfg] = mvpalab_mvpa(cfg,fv);
cfg.classmodel.permlab = true;
[peracc,cfg] = mvpalab_mvpa(cfg,fv);

%% Sliding filter analysis - Generate diffMaps:

if strcmp(cfg.stats.pmetric,'cr')
    
    acc = acc.cr;
    peracc = peracc.cr;
    
    permaps = permaps.cr;
    accmap = accmap.cr;
    
    mvpalab_mkdir([cfg.location filesep 'stats'...
        filesep 'cr']);
elseif strcmp(cfg.stats.pmetric,'auc')
    
    acc = acc.auc;
    peracc = peracc.auc;
    
    permaps = permaps.auc;
    accmap = accmap.auc;
    
    mvpalab_mkdir([cfg.location filesep 'stats'...
        filesep 'auc']);
end

[diffMap,perdiffMap,cfg] = mvpalab_gendiffmap(...
    cfg,acc,accmap,peracc,permaps);

%% Permutation test:
stats = mvpalab_permtest(cfg,diffMap,perdiffMap);

%% Save stats and cfg structure:
save([cfg.location filesep...
    'stats' filesep cfg.stats.pmetric filesep ...
    'stats.mat'],'stats','-v7.3');
save([cfg.location filesep 'cfg.mat'],'cfg','-v7.3');

