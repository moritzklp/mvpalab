function [x,y,t,auc,cr,cm] = roc_classifier(X,Y,tp,cfg,strpar)
%SVM_CLASSIFIER This function returns the accuracy of the classifier in a
%time-resolved way.

% x = x-coordinates of the ROC curve.
% y = y-coordinates of the ROC curve.
% t = 
% auc = Area Under the Curve value.
% cr = Correct rate.
% cm = Confusion matrix.


%% Kfold validation loop:
if ~cfg.analysis.tempgen
    predicted_labels = true(strpar.NumObservations,1);
    predicted_scores = ones(strpar.NumObservations,2);
else
    for i = 1 : cfg.analysis.ntp
        predicted_labels{i} = true(strpar.NumObservations,1);
        predicted_scores{i} = ones(strpar.NumObservations,2);
    end
end
for k = 1 : strpar.NumTestSets
    % Train and test data:
    train_X = X(strpar.training(k),:,cfg.analysis.tpoints(tp));
    train_Y = Y(strpar.training(k));
    test_X = X(strpar.test(k),:,cfg.analysis.tpoints(tp));
    test_Y = Y(strpar.test(k));
    
    % Data normalization if needed:
    if cfg.analysis.datanorm == 3
        train_X = zscore(train_X,[],1);
        test_X = zscore(test_X,[],1);
    elseif cfg.analysis.datanorm == 4
        [train_X, test_X, norm_params] = std_norm(train_X,test_X);
    end
    
    % Permute labels if needed:
    if cfg.analysis.permlab
        train_Y = train_Y(randperm(length(train_Y)));
    end
    
    %% Feature selection if needed:
    if cfg.analysis.pca.flag || cfg.analysis.pls.flag
        [train_X,test_X,params] = ...
            feature_selection(train_X,train_Y,test_X,test_Y,cfg);
    end
    
    %% Train and test SVM model
    % Hyperparameter optimization if needed:
    if cfg.analysis.optimize.flag
        mdlSVM = fitcsvm(train_X,train_Y,...
            'OptimizeHyperparameters',cfg.analysis.optimize.params,...
            'HyperparameterOptimizationOptions',cfg.analysis.optimize.opt);
    else
        mdlSVM = fitcsvm(train_X,train_Y);
    end
    
    % Temporal generalization if needed:
    if cfg.analysis.tempgen
        for tp2 = 1 : cfg.analysis.ntp
            test_X = X(strpar.test(k),:,cfg.analysis.tpoints(tp2));
            
            % Data normalization if needed:
            if cfg.analysis.datanorm == 3
                test_X = zscore(test_X,[],1);
            elseif cfg.analysis.datanorm == 4
                [~,test_X] = std_norm([],test_X,norm_params);
            end
            
            % Project new test set in the PC space if needed
            if cfg.analysis.pca.flag; test_X = project_pca(test_X,params,cfg); end
            
            % Predict labels:
            [labels,scores] = predict(mdlSVM,test_X);
            acc(k,tp2) = sum(test_Y == labels)/length(test_Y);
            
            % Update label and score vectors:
            predicted_labels{tp2}(strpar.test(k)) = labels;
            predicted_scores{tp2}(strpar.test(k),:) = scores;
            
        end
    else
        % Test classifier:
        [labels,scores] = predict(mdlSVM,test_X);
        acc(k) = sum(test_Y == labels)/length(test_Y);
        predicted_labels(strpar.test(k)) = labels;
        predicted_scores(strpar.test(k),:) = scores;
    end
end


%% Calculate the ROC curve for temporal generalization:
if cfg.analysis.tempgen
    for tp2 = 1 : cfg.analysis.ntp
        %% Compute confusion matrix:
        cm{tp2} = confusionmat(Y,predicted_labels{tp2});
        %% Receiver operating characteristic (ROC curve)
        [x{tp2},y{tp2},t{tp2},auc(tp2)] = ...
            perfcurve(Y,predicted_scores{tp2}(:,mdlSVM.ClassNames),1);
        % Correct rate:
        cr = mean(acc);
    end
else
    % Compute confusion matrix:
    cm = confusionmat(Y,predicted_labels);
    % Receiver operating characteristic (ROC curve)
    [x,y,t,auc] = perfcurve(Y,predicted_scores(:,mdlSVM.ClassNames),1);
    % Correct rate:
    cr = mean(acc);
end

%% EASIER FOR READING BUT LESS FLEXIBLE:
% tic;
% mdlCVSVM = crossval(mdlSVM,'CVPartition',strpar);
% cr = 1 - kfoldLoss(mdlCVSVM);
%
% %% Predicted labels:
% [predicted_labels,predicted_scores] = kfoldPredict(mdlCVSVM);
%
% %% Compute confusion matrix:
% cm = confusionmat(Y,predicted_labels);
%
% %% Receiver operating characteristic (ROC curve)
% [x,y,t,auc] = perfcurve(Y,predicted_scores(:,mdlSVM.ClassNames),1);
% toc;
end