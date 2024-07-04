function rdm = mvpalab_computerdm(cfg, X, dims)
%% MVPALAB_RDM
%
%  This function computes the Representational Dissimilarity Matrix.
%
%%  INPUT:
%
%  - {struct} - cfg:
%    Configuration structure.
%
%  - {2D-matrix} - X:
%    Data matrix for an individual subject containing all the trials and
%    conditions. [trials x channels]
%
%  - {1D-array} - dims:
%    Dimensions of the subject_betas (only for fMRI). [rows, columns] = [regions, runs]
%
%
%%  OUTPUT:
%
%  - {2D-matrix} - rdm:
%    Representational Dissimilarity Matrix:
%    [trials x trials] or [n_conditions x n_conditions]
%

if cfg.rsa.normrdm
    X = zscore(X, [], [1, 2]);
end

if strcmp(cfg.rsa.distance, 'pearson')
    rdm = 1 - corrcoef(X');
elseif strcmp(cfg.rsa.distance, 'euclidean')
    rdm = pdist2(X, X, 'euclidean');
elseif strcmp(cfg.rsa.distance, 'mahalanobis')
    covMatrix = calc_covariance_matrix(X);
    
    %rdm = pdist2(X, X, 'mahalanobis', covMatrix);
    rdm = mahalanobis_distance(X, covMatrix); 

elseif strcmp(cfg.rsa.distance, 'cvmahalanobis')
    if strcmp(cfg.analysis, 'rsa-roi')
        k = cfg.rsa.cvFolds;
        numRuns = dims(2);
        trialsPerRun = dims(1);

        if k < numRuns
            k = numRuns;
            warning('The number of folds must be greater than or equal to the number of runs, setting the number of folds to %d.', k);
        end

        if mod(k, numRuns) ~= 0
            k = k - mod(k, numRuns);
            warning('The number of folds is not a multiple of the number of runs, reducing the number of folds to %d.', k);
        end

        runs = cell(numRuns, 1); % Initialize the cell array to hold run data
        for runIndex = 1:numRuns
            % Calculate the start and end indices for slicing X
            startIndex = (runIndex - 1) * trialsPerRun + 1;
            endIndex = runIndex * trialsPerRun;
            
            % Check if X has enough rows for the current run
            if endIndex > size(X, 1)
                error('Not enough data for the specified number of trials per run.');
            end
            
            % Slice X for the current run and assign it to the runs cell array
            runs{runIndex} = X(startIndex:endIndex, :);
        end

        
    else
        numRuns = cfg.rsa.cvFolds;
        runs = cell(numRuns, 1);
        trialsPerRun = ceil(size(X, 1) / numRuns);  % Calculate trials per run to handle non-even division
        for i = 1:numRuns
            startIdx = (i - 1) * trialsPerRun + 1;
            endIdx = min(i * trialsPerRun, size(X, 1));  % Ensure we do not exceed the number of trials
            runs{i} = X(startIdx:endIdx, :);
        end
    end

    % Initialize the RDM matrix
    rdm = zeros(numRuns, size(X, 1), size(X, 1));

    % Iterate over each run for the test set
    for testRunIndex = 1:numRuns
        testIdx = (testRunIndex - 1) * trialsPerRun + 1:min(testRunIndex * trialsPerRun, size(X, 1));
        testData = runs{testRunIndex};

        % Combine the remaining runs for the training set
        trainData = [];
        trainIdx = [];
        for trainRunIndex = 1:numRuns
            if trainRunIndex == testRunIndex
                continue; % Skip if the test run and train run are the same
            end
            currentRunData = runs{trainRunIndex};
            trainData = [trainData; currentRunData];
            startIdx = (trainRunIndex - 1) * trialsPerRun + 1;
            endIdx = min(trainRunIndex * trialsPerRun, size(X, 1));
            trainIdx = [trainIdx, startIdx:endIdx];
        end

        covTrain = calc_covariance_matrix(trainData);

        % Compute Mahalanobis distances in bulk
        %distMatrix = pdist2(X, X, 'mahalanobis', covTrain);
        distMatrix = mahalanobis_distance(X, covTrain); 
        % save dist matrix in first fold of rdm
        rdm(testRunIndex, :, :) = distMatrix;
    end

    % Average the RDMs across the folds
    rdm = squeeze(mean(rdm, 1));
else
    rdm = zeros(size(X, 1), size(X, 1));
    % print error message:
    error('Distance measure not recognized.');
end
end

function covMatrix = calc_covariance_matrix(data)
    [t, n] = size(data);
    meanx = mean(data);
    X = data - meanx(ones(t, 1), :);
    
    % compute sample covariance matrix
    sample = (1 / t) * (X' * X);
    
    % ensure symmetry
    sample = (sample + sample') / 2;
    
    % compute prior
    prior = diag(diag(sample));
    
    % compute shrinkage parameters
    d = 1 / n * norm(sample - prior, 'fro')^2;
    y = X.^2;
    r2 = 1 / n / t^2 * sum(sum(y' * y)) - 1 / n / t * sum(sum(sample.^2));
    
    % compute the estimator
    shrinkage = max(0, min(1, r2 / d));
    covMatrix = shrinkage * prior + (1 - shrinkage) * sample;
    
    covMatrix = pinv(covMatrix);
    
    covMatrix = (covMatrix + covMatrix') / 2;
    
    % test that covMatrix is in fact PD. if it is not so, then tweak it just a bit.
    p = 1;
    k = 0;
    while p ~= 0
        [~, p] = chol(covMatrix);
        k = k + 1;
        if p ~= 0
            mineig = min(eig(covMatrix));
            covMatrix = covMatrix + (-mineig * k.^2 + eps(mineig)) * eye(size(covMatrix, 1));
            warning('Added %d to covariance matrix due to numerical issues.', -mineig * k.^2 + eps(mineig));
        end
    end
    
    % Check if the covariance matrix is positive definite
    tol = 1e-3;
    if any(eig(covMatrix) < -tol)
        warning('Covariance matrix may not be positive definite due to numerical issues.');
    end
    
    % check if square
    if size(covMatrix, 1) ~= size(covMatrix, 2)
        warning('Covariance matrix is not square.');
    end
    
    % check if symmetric
    if ~issymmetric(covMatrix)
        warning('Covariance matrix is not symmetric.');
    end
    
    % check if number of columns in X matches the covariance matrix
    if size(X, 2) ~= size(covMatrix, 1)
        warning('Number of columns in X does not match the covariance matrix.');
    end
end

function rdm = mahalanobis_distance(data, cov_matrix)
    % mahalanobis_distance Calculate the Mahalanobis distance between all pairs of points in a data matrix
    %   D = mahalanobis_distance(data, cov_matrix)
    %   data: an m-by-n matrix where each row is a data point
    %   cov_matrix: a precomputed n-by-n covariance matrix
    %   D: an m-by-m matrix where each element D(i,j) is the Mahalanobis distance between data(i,:) and data(j,:)

    % Ensure data is a matrix
    if size(data, 1) == 1
        data = data';
    end

    % Calculate the mean of the data
    mean_data = mean(data, 1);

    % Center the data
    centered_data = data - mean_data;

    % Calculate the inverse of the covariance matrix
    inv_cov_matrix = pinv(cov_matrix);

    % Initialize the distance matrix
    n = size(data, 1);
    rdm = zeros(n, n);

    % Calculate the Mahalanobis distance for all pairs of points
    for i = 1:n
        for j = i:n
            diff = centered_data(i, :) - centered_data(j, :);
            rdm(i, j) = sqrt(diff * inv_cov_matrix * diff');
            rdm(j, i) = rdm(i, j); % Symmetric distance
        end
    end
end
