%exp4
% Increasing the class separbility
clear all

dim= 3;
sets = 10;

% class separability multiplication factors
c1 = 1;
%c2 = a^x;

% monotonically increasing function f(x) = a^x with a > 1
a = 1.15;

% Sets
Nd = 500;

% covariance matrix
sigma = eye(dim);

for x = 1:sets
            for i = 1:10
            
            % means class 1 and 2
            u1 = ones([dim, 1]) * c1;
            u2 = ones([dim, 1]) * a^x;
            
            % create train set
            train1 = mvnrnd(u1, sigma, Nd);
            train2 = mvnrnd(u2, sigma, Nd);
            trainset = [train1; train2];

            % create test set
            test1 = mvnrnd(u1, sigma, Nd);
            test2 = mvnrnd(u2, sigma, Nd);
            testset = [test1; test2];

            % Create labels train and test sets
            labels_trainset = [zeros(Nd,1) ; ones(Nd,1)];
            labels_testset = [zeros(Nd,1); ones(Nd,1)];

            % Train classifier
            gaussian_classifier = fitcnb(trainset, labels_trainset);
            
            % Test with testset as test error
            y_pred_test = predict(gaussian_classifier, testset);
            % Calculate test error for trial i
            error_test(i) = sum(y_pred_test ~= labels_testset)/numel(y_pred_test);

            end
        % Calculate average test error
        error_test_avg(x) = mean(error_test);
end

for x = 1:sets
    distance(x) = a^x-1;
end

figure()
plot(distance, error_test_avg)
xlabel('Mahalanobis Distance')
ylabel('Classification Error (%)')
title('Classification Error vs Mahalanobis inter-means dist.')
legend(sprintf('Dimensions: %d, sets: %d', dim, sets))
            
            


