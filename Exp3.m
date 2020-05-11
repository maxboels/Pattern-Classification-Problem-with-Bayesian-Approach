% exp3

clear all
% dimensions
dim = [5, 10, 15];

%number of samples from training set
Nd = [3,5,10,20,50,100,200];

% Aconstant per dimension
% Adjust based on trial and error
c1 = [1, 1, 1];
c2 = [2.4, 2, 1.8];

Nd_TrueError = 500;

for n = 1:length(dim)
 for d = dim(n)
     for s = 1:length(Nd)
        for sample = Nd(s)
            for i = 1:10
                % covariance matrix
                sigma = eye(d);

                % means class 1 and 2
                u1 = ones([d, 1]) * c1(n);
                u2 = ones([d, 1]) * c2(n);

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

%                 % Test with trainset as Design error
%                 y_pred_design = predict(gaussian_classifier, trainset);
%                 % Calculate design error for trial i
%                 error_design(i) = sum(y_pred_design ~= labels_trainset)/numel(y_pred_design);

                % Test with testset as test error
                y_pred_test = predict(gaussian_classifier, testset);
                % Calculate test error for trial i
                error_test(i) = sum(y_pred_test ~= labels_testset)/numel(y_pred_test);
            end
%           % Calculate average design error
%           error_design_avg(s) = mean(error_design);
          % Calculate average test error
          error_test_avg(s) = mean(error_test);
        end
     end
%     error_design_avg_dim(n,:) = error_design_avg;
    error_test_avg_dim(n,:) = error_test_avg;
 end 
end


% figure()
% for n = 1:length(dim)
%  for d = dim(n)
%     plot(Nd, error_design_avg_dim(n,:))
%     hold on
%  end
% end
% 
% xlabel('Number of samples')
% ylabel('Classification Error')
% title('Classification Error Design sets By Dimensions')
% legend(sprintf('Number of dimensions: %d', dim(1)), ...
%        sprintf('Number of dimensions: %d', dim(2)), ...
%        sprintf('Number of dimensions: %d', dim(3)))
   
figure()
for n = 1:length(dim)
 for d = dim(n)
    plot(Nd, error_test_avg_dim(n,:))
    hold on
 end
end

TrueError = 0.06;
xlabel('Number of samples')
ylabel('Classification Error (%)')
yline(0.06, '--');
title('Classification Error Test sets By Dimensions')
legend(sprintf('Number of dimensions: %d', dim(1)), ...
       sprintf('Number of dimensions: %d', dim(2)), ...
       sprintf('Number of dimensions: %d', dim(3)), ...
       sprintf('True Error for 500 samples at: %.2f', TrueError))

