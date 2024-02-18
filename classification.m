% *Exercise 2: Classification* 

clear all;
close all;
addpath ('C:\Users\Amineh\Desktop\ESTIMATION\Assignment\Functions')

N=1000; %set it at a suitable amount

Student = 'Amineh Yazdizadeh Baghini';
Matriculation = '0000998863';
[FeatureSet] = ClassifyThose(N,Student,Matriculation);

Utrain  = FeatureSet.Utrain;
Ytrain  = FeatureSet.Ytrain;
Utest   = FeatureSet.Utest;
Ytest   = FeatureSet.Ytest;
% *2.1Understand the feature set in space*
% our input u is in $R^2$and we have 4 class.
% 
% to understand the feature set in space and potentially define a refined feature 
% set indicates that we may want to explore and visualize the distribution of 
% your features. Plotting can be a helpful tool for gaining insights into the 
% data

% Scatter plot of the feature space
figure;
gscatter(Utrain(:,1), Utrain(:,2), Ytrain, 'gbry')
xlabel('Feature1');
ylabel('Feature2');
legend("class 1","class 2","class 3","class 4")
title('Training Data Feature Space');

scatter3(Utrain(:, 1), Utrain(:, 2), Ytrain, [], Ytrain, 'filled');
title('Training Set Features');
xlabel('Feature 1');
ylabel('Feature 2');
zlabel('Output (Ytrain)');
grid on;
colorbar;
%% 
% this plot shows clearly that this is a 4 class clssification problem.
% 
% 
% 
% In order to obtain better posed matrices, we can use standardization or normalization 
% techniques. Standardization scales the features so that they have a mean of 
% 0 and a standard deviation of 1, while normalization scales the features to 
% a specific range, often between 0 and 1. I simply rescale the feature set in 
% the following way:

% Standardize the training set features
[Utrain, mu, sigma] = zscore(Utrain);

% Apply the same transformation to the testing set features
Utest = (Utest - mu) ./ sigma;
%% 
% The testing set features (|Utest|) are then standardized using the mean and 
% standard deviation obtained from the training set.
% 
% Make sure to apply the same transformation to both the training and testing 
% sets to maintain consistency. This process helps in ensuring that the features 
% are on a similar scale, which can be important for certain machine learning 
% algorithms, especially those that rely on distance metrics.
% 
% 
% *2.2 Define the required classification algorithm*
% For a classification problem with 4 classes, we can use the Support Vector 
% Machine (SVM).
% 
% Support Vector Machines are supervised learning models that analyze and classify 
% data points into different classes. SVM works by finding the hyperplane that 
% best separates the data into distinct classes. The optimal hyperplane is the 
% one that maximizes the margin, which is the distance between the hyperplane 
% and the nearest data points from each class.
% 
% SVM can handle both linear and non-linear classification tasks through the 
% use of different kernel functions. The decision boundary is determined by support 
% vectors, which are the data points that lie closest to the hyperplane.

% Train the classifier on the training set
svmModel = fitcecoc(Utrain, Ytrain);

% Predict on the training set
trainPredictions = predict(svmModel, Utrain);

% Compute training classification accuracy and error
trainAccuracy = sum(trainPredictions == Ytrain) / numel(Ytrain);
trainError = 1 - trainAccuracy;

% Display training results
disp(['Training Classification Accuracy: ', num2str(trainAccuracy * 100), '%']);
disp(['Training Classification Error: ', num2str(trainError * 100), '%']);

% Predict on the test set
testPredictions = predict(svmModel, Utest);

% Compute test classification accuracy and error
testAccuracy = sum(testPredictions == Ytest) / numel(Ytest);
testError = 1 - testAccuracy;

% Display test results
disp(['Test Classification Accuracy: ', num2str(testAccuracy * 100), '%']);
disp(['Test Classification Error: ', num2str(testError * 100), '%']);

% Plot results on the test set
figure;
subplot(2, 1, 1);
scatter(Utest(:, 1), Utest(:, 2), 50, Ytest, 'filled');
title('Actual Classes (Test Set)');
xlabel('Feature 1');
ylabel('Feature 2');
colormap('jet');
colorbar;

subplot(2, 1, 2);
scatter(Utest(:, 1), Utest(:, 2), 50, testPredictions, 'filled');
title('Predicted Classes (Test Set)');
xlabel('Feature 1');
ylabel('Feature 2');
colormap('jet');
colorbar;
%% 
% A scatter plot visualize the actual and predicted classes in the test set.
% 
% The classifier is trained using the |fitcecoc| function on the training set.
% 
% Predictions are made on both the training set and the test set.
% 
% Classification errors for training set is 0,142% ,and for test set is 1%.
% 
% Training Set Classification Error (0.142%):
% 
% The low training set classification error suggests that the classifier has 
% learned the patterns in the training data effectively.
% 
% A very low error indicates that the classifier is able to accurately predict 
% the classes of instances it has seen during training.
% 
% Test Set Classification Error (1%):
% 
% The test set classification error of 1% is also relatively low, indicating 
% good generalization performance.
% 
% A low error on the test set suggests that the classifier is not overfitting 
% to the training data and can generalize well to new, unseen instances.
% 
% Overall, a low training set classification error combined with a similarly 
% low test set classification error is a positive outcome. It indicates that your 
% classifier is likely performing well and is capable of making accurate predictions 
% on new data.
% 
% 
%