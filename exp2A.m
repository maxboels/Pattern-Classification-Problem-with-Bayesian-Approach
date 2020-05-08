% exp2
% In this experiment we shall investigate the effect of the size of test set on the reliability of the
% empirical error count estimator and we will also learn how to classify test and training samples.

% covariance matrix
sigma = [2 1; 1 2];

% means class 1 and 2
u1 = [1; 2];
u2 = [4; 5];

%number of samples from test patterns
Nt = [3,5,10,50,100];

for s = 1:length(Nt)
 for sample = Nt(s)
    % create 10 independent test sets
    for i = 1:10
    % create training set
    train1 = mvnrnd(u1, sigma, sample);
    train2 = mvnrnd(u2, sigma, sample);
    trainset = [train1; train2];

    % generate 10 independent test sets
    test1 = mvnrnd(u1, sigma, 100);
    test2 = mvnrnd(u2, sigma, 100);
    testset = [test1; test2];

    % Create labels for training (sample size) and test set.
    class1 = zeros(sample,1);
    class2 = ones(sample,1);
    labels_trainset = [class1 ; class2];
    labels_testset = [zeros(100,1); ones(100,1)];

    % Train classifier with trainset
    gaussian_classifier = fitcnb(trainset, labels_trainset);

    % Test with testset as test error
    y_pred_test = predict(gaussian_classifier, testset);
    % Calculate test error for trial i
    error_test(i) = sum(y_pred_test ~= labels_testset)/numel(y_pred_test);
    end

    % Calculate average test error for each sample size.
    error_test_avg(s) = mean(error_test);
    % the variance is normalized by the number of observations-1 by default.
    error_test_var(s) = var(error_test);
  end
end

figure()
plot(Nt,error_test_avg)
xlabel('Samples size')
ylabel('Mean Classification Error')
title('Mean Classification Error vs Samples size')
hold off

figure()
plot(Nt, error_test_var)
xlabel('Sample size')
ylabel('Variance of Mean Classification Error')
title('Variance of Mean Classification Error vs sample size')
hold off


% comment on the results:
