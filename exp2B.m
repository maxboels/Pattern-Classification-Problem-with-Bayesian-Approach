% exp2B

% covariance matrix
sigma = [2 1; 1 2];

% means class 1 and 2
u1 = [1; 2];
u2 = [4; 5];

% Dataset = 100 observations per class?????????
% Nd training and Nt testing sample split.
Nd = [3, 5, 10, 50, 75, 90];
Nt = [97, 95, 90, 50, 25, 10];


for s = 1:length(Nd)
 for sample = Nt(s)

    for i = 1:10
    % create training set
    train1 = mvnrnd(u1, sigma, Nd(s));
    train2 = mvnrnd(u2, sigma, Nd(s));
    trainset = [train1; train2];

    % generate 10 independent test sets
    test1 = mvnrnd(u1, sigma, Nt(s));
    test2 = mvnrnd(u2, sigma, Nt(s));
    testset = [test1; test2];

    % Create labels for train and test sets
    class1_train = zeros(Nd(s),1);
    class2_train = ones(Nd(s),1);
    class1_test = zeros(Nt(s),1);
    class2_test = ones(Nt(s),1);
    labels_trainset = [class1_train ; class2_train];
    labels_testset = [class1_test ; class2_test];

    % Train classifier
    gaussian_classifier = fitcnb(trainset, labels_trainset);

    % Test with trainset as Design error
    y_pred_design = predict(gaussian_classifier, trainset);
    % Calculate design error for trial i
    error_design(i) = sum(y_pred_design ~= labels_trainset)/numel(y_pred_design);

    % Test with testset as test error
    y_pred_test = predict(gaussian_classifier, testset);
    % Calculate test error for trial i
    error_test(i) = sum(y_pred_test ~= labels_testset)/numel(y_pred_test);


    end


% Calculate average design and test error for each sample size split.
error_design_avg(s) = mean(error_design);
error_test_avg(s) = mean(error_test);

end
end

figure()
% create a table with design error and test error:
plot(Nd,error_design_avg)
title('Design Error')
xlabel('Design sample size')
hold off

figure()
% create a table with design error and test error:
plot(Nt,error_test_avg)
title('Test Error')
xlabel('Test sample size')



