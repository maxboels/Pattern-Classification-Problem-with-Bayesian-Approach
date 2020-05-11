%exp4
% Increasing the class separbility
clear all

dim= 3;
sets = 5;

% class separability multiplication factors
c1 = 1;
%c2 = a^x;

% monotonically increasing function f(x) = a^x with a > 1
a = 1.3;

% Sets
Nd = [3, 5, 15, 30, 100];

% covariance matrix
sigma = eye(dim);

for x = 1:sets
    for s = 1:length(Nd)
        for sample = Nd(s)
            for i = 1:10
            
            % means class 1 and 2
            u1 = ones([dim, 1]) * c1;
            u2 = ones([dim, 1]) * a^x;
            
            % create train set
            train1 = mvnrnd(u1, sigma, sample);
            train2 = mvnrnd(u2, sigma, sample);
            trainset = [train1; train2];

            % create test set
            test1 = mvnrnd(u1, sigma, sample);
            test2 = mvnrnd(u2, sigma, sample);
            testset = [test1; test2];

            % Create labels train and test sets
            labels_trainset = [zeros(sample,1) ; ones(sample,1)];
            labels_testset = [zeros(sample,1); ones(sample,1)];

            % Train classifier
            gaussian_classifier = fitcnb(trainset, labels_trainset);
            
            % Test with testset as test error
            y_pred_test = predict(gaussian_classifier, testset);
            % Calculate test error for trial i
            error_test(i) = sum(y_pred_test ~= labels_testset)/numel(y_pred_test);

            end
        % Calculate average test error
        error_test_avg(s) = mean(error_test);
        end
    end
error_test_avg_dim(x,:) = error_test_avg;
end

figure()
for n = 1:length(Nd)
    plot(Nd, error_test_avg_dim(n,:))
    hold on
end

xlabel('Number of samples')
ylabel('Classification Error (%)')
x = 1:sets;
title('Classification Error vs Sample size by distance between Means')
legend(sprintf('Distance between means: %.2f', a^x(1)-c1), ...
       sprintf('Distance between means: %.2f', a^x(2)-c1), ...
       sprintf('Distance between means: %.2f', a^x(3)-c1), ...
       sprintf('Distance between means: %.2f', a^x(4)-c1), ...
       sprintf('Distance between means: %.2f', a^x(5)-c1))
            
            


