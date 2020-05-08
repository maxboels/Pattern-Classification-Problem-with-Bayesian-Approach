

sigma = [2 1; 1 2];

u1 = [1; 2]; % class 1
u2 = [4; 5]; % class 2

%number of samples from training set
Nd = [3,5,10,50,100];

for n = 1:length(Nd)
 for sample = Nd(n)
    for i = 1:10
    % create training set
    train1 = mvnrnd(u1, sigma, sample);
    train2 = mvnrnd(u2, sigma, sample);
    trainset = [train1; train2];

    % create test set
    test1 = mvnrnd(u1, sigma, 100);
    test2 = mvnrnd(u2, sigma, 100);
    testset = [test1; test2];
    
    % Create labels
    class1 = zeros(sample,1);
    class2 = ones(sample,1);
    labels = [class1 ; class2];
    labels_testset = [zeros(100,1); ones(100,1)];

    % Train classifier
    gaussian_classifier = fitcnb(trainset, labels);
    
    % Test with trainset as Design error
    y_pred_design = predict(gaussian_classifier, trainset);
    % Calculate design error for trial i
    error_design(i) = sum(y_pred_design ~= labels)/numel(y_pred_design);
    
    % Test with testset as test error
    y_pred_test = predict(gaussian_classifier, testset);
    % Calculate test error for trial i
    error_test(i) = sum(y_pred_test ~= labels_testset)/numel(y_pred_test);
    end
    
  % Calculate average design error
  error_design_avg(n) = mean(error_design);
  % Calculate average test error
  error_test_avg(n) = mean(error_test);
  end
end


plot(Nd, error_design_avg)
hold on
plot(Nd, error_test_avg)
legend('e_d_e_s_i_g_n_ avg','e_t_e_s_t_ avg')
xlabel('Number of samples')
ylabel('Classification Error')
title('Classification Error Design and Test sets')

% Compare with theoritical error.
