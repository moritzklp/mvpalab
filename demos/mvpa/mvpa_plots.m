clc;
clear;

%% Plot configuration:
graph = mvpalab_plotinit();

%% Mean accuracy plot (no statistical significance)

% Load results:
load results/macc/result.mat

% Axis limits:
graph.xlim = [-200 1500];
graph.ylim = [.3 .95];

% Axes labels and titles:
graph.xlabel = 'Time (ms)';
graph.ylabel = 'Classifier performance';
graph.title = 'Demo plot (no statistical significance)';

% Smooth results:
graph.smoothdata = 1; % (1 => no smoothing)

% Plot results:
figure;
hold on
mvpalab_plotcr(graph,cfg,result);

%% Mean accuracy plot (statistical significance)

% Load results and and statistics:
load results/macc/result.mat
load results/macc/stats.mat

% Plot significant clusters (above and below chance):
graph.stats.above = true;
graph.stats.below = true;

% Significant indicator:
graph.sigh = .4;

% Title:
graph.title = 'Demo plot (statistical significance)';

% Plot results:
figure;
hold on
mvpalab_plotcr(graph,cfg,result,stats);