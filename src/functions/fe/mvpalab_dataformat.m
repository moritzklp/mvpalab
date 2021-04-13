function [cfg,EEG] = mvpalab_dataformat(cfg,input)

% Extract field names:
fields = fieldnames(input);

% Extract data:
data = input.(fields{1});

if isfield(data,'fsample')
    %% Fieldtrip:
    % Store sampling frequency:
    cfg.fs = data.fsample;
    
    % Prepare data format:
    EEG.srate = data.fsample;
    EEG.trials = length(data.trial);
    EEG.nbchan = length(data.label);
    EEG.pnts = size(data.trial{1},2);
    EEG.times = data.time{1} * 1000;
    for trial = 1 : length(data.trial)
        EEG.data(:,:,trial) = data.trial{trial};
    end
    
else
    %% EEGlab:
    % Store sampling frequency:
    cfg.fs = data.srate;
    % Generate data file:
    EEG = data;
end
end

