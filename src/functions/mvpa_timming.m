function [ cfg ] = mvpa_timming( cfg )
%MVPA_TIMMING This function configures the data in the cfg structure for 
%the mvpa analysis.


%% Find closest points in the time vector:

[~,cfg.startidx] = min(abs(cfg.times-cfg.tpstart));
[~,cfg.endidx] = min(abs(cfg.times-cfg.tpend));

%% Adjust sizes and timepoints:

cfg.tpoints = (cfg.startidx:cfg.tpsteps:cfg.endidx);
cfg.ntp = length(cfg.tpoints);
cfg.times = cfg.times(cfg.startidx:cfg.tpsteps:cfg.endidx);

end

