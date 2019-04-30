%% Support Vector Machines 
clc; clear all; close all; 
warning('off','all'); 
% Load the datasets 
svm_data = load('data_ps3_2.mat'); 
% Training Data 
set1_train = svm_data.set1_train; % Set 1
set2_train = svm_data.set2_train; % Set 2
set3_train = svm_data.set3_train; % Set 3
set4_train = svm_data.set4_train; % Set 4

% Testing Data
set1_test = svm_data.set1_test; % Set 1
set2_test = svm_data.set2_test; % Set 2
set3_test = svm_data.set3_test; % Set 3
set4_test = svm_data.set4_test; % Set 4


%% Question 1: 
% For the first three datasets, consider the linear, second 
% order polynomial, Gaussian of standard deviation 1 kernels 

%% Part 1: 

% Train first 3 datasets, plot the decision boundary and print the test
% errors 

svm_test(@Klinear,[],1000,set1_train,set1_test); 
svm_test(@Klinear,[],1000,set2_train,set2_test); 
svm_test(@Klinear,[],1000,set3_train,set3_test); 

svm_test(@Kpoly,2,1000,set1_train,set1_test); 
svm_test(@Kpoly,2,1000,set2_train,set2_test); 
svm_test(@Kpoly,2,1000,set3_train,set3_test); 

svm_test(@Kgaussian,1,1000,set1_train,set1_test); 
svm_test(@Kgaussian,1,1000,set2_train,set2_test); 
svm_test(@Kgaussian,1,1000,set3_train,set3_test); 

%% Part 2 

% Train 4th dataset with a linear polynomial of degree 2, and Gaussian of
% standard deviation 1.5 kernels 
TestError = zeros(3,1);

svm_linear4 = svm_train(set4_train,@Klinear,[],1000); 
y_linear4 = sign(svm_discrim_func(set4_test.X,svm_linear4));
errors_linear = find(y_linear4 ~= set4_test.y);
TestError(1) = length(errors_linear)/length(set4_test.y);
fprintf('Linear SVM: %g of 4th test examples  were misclassified.\n',...
    length(errors_linear)/length(set4_test.y));

svm_poly4 = svm_train(set4_train,@Klinear,2,1000); 
y_poly4 = sign(svm_discrim_func(set4_test.X,svm_poly4));
errors_poly = find(y_poly4 ~= set4_test.y);
TestError(2) = length(errors_poly)/length(set4_test.y);
fprintf('Polynomial SVM: %g of 4th test examples  were misclassified.\n',...
    length(errors_poly)/length(set4_test.y));

svm_gaussian4 = svm_train(set4_train,@Kgaussian,1.5,1000); 
y_gaussian4 = sign(svm_discrim_func(set4_test.X,svm_gaussian4));
errors_gaussian = find(y_gaussian4 ~= set4_test.y);
TestError(3) =  length(errors_gaussian)/length(set4_test.y);
fprintf('Gaussian SVM: %g of 4th test examples  were misclassified.\n',...
    length(errors_gaussian)/length(set4_test.y));
Kernel = {'Linear';'Polynomial of degree 2';'Gaussian of std 1.5'}; 

% The test errors of 4th dataset trained on different kernels is as below: 
t = table(TestError,Kernel);
display(t);

%% Bayes Classifier 





