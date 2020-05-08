

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


% KNN Classifier

sigma = [2 1; 1 2];

u1 = [1; 2]; % class 1
u2 = [4; 5]; % class 2

%number of samples from training set
Nd = [3,5,10,50,100];

% number of nearest neighbours
K = 1:2:51;

for j = 1:length(K)
for k = K(j)

    for n = 1:length(Nd)
     for sample = Nd(n)
        for i = 1:10
        % create training set
        train1 = mvnrnd(u1, sigma, sample); % class 1
        train2 = mvnrnd(u2, sigma, sample); % class 2
        trainset = [train1; train2];

        % create test set
        test1 = mvnrnd(u1, sigma, 100); % class 1
        test2 = mvnrnd(u2, sigma, 100); % class 2
        testset = [test1; test2];

        % Create labels
        class1 = zeros(sample,1);
        class2 = ones(sample,1);
        labels_trainset = [class1 ; class2];
        labels_testset = [zeros(100,1); ones(100,1)];

        % train classifier KNN
        Mdl = fitcknn(trainset, labels_trainset,'NumNeighbors',k ,...
        'NSMethod','exhaustive','Distance','minkowski',...
        'Standardize',1);

        % Test with trainset as Design error
        y_pred_design = predict(Mdl, trainset);
        % Calculate design error for trial i
        knn_error_design(i) = sum(y_pred_design ~= labels_trainset)/numel(y_pred_design);

        % Test with testset as test error
        y_pred_test = predict(Mdl, testset);
        % Calculate test error for trial i
        knn_error_test(i) = sum(y_pred_test ~= labels_testset)/numel(y_pred_test);
        end

      % Calculate average design error
      knn_error_design_avg(n) = mean(knn_error_design);
      % Calculate average test error
      knn_error_test_avg(n) = mean(knn_error_test);
      end
    end

  % Calculate average design error for k
  all_error_design_avg_K(j,:) = knn_error_design_avg;
  % Calculate average test error for k
  all_error_test_avg_K(j,:) = knn_error_test_avg;
end 
end

%minimum error for all k
error_design_avg_K = min(all_error_design_avg_K(2:end,:))
error_test_avg_K = min(all_error_test_avg_K(2:end,:))

% figure()
% plot(Nd, error_design_avg_K)
% xlabel('Number of K nearest neighbours')
% ylabel('Classification Error')
% legend('Min. Design error for all k')
% title('Classification Error Design set vs k-nearest neighbours')
% hold off
% 
% figure()
% plot(Nd, error_test_avg_K)
% xlabel('Number of K nearest neighbours')
% ylabel('Classification Error')
% legend('Min. Test error for all k')
% title('Classification Error Test set vs k-nearest neighbours')
% hold off

% Compare with theoritical error.
figure()
plot(Nd, error_design_avg)
hold on
plot(Nd, error_test_avg)
hold on
plot(Nd, error_design_avg_K)
hold on
plot(Nd, error_test_avg_K)
legend('e_d_e_s_i_g_n_ avg','e_t_e_s_t_ avg', 'knn e_d_e_s_i_g_n_ best k', 'knn e_t_e_s_t_ best k')
xlabel('Number of samples')
ylabel('Classification Error')
title('Classification Error Design and Test sets Gaussian v KNN')
hold off