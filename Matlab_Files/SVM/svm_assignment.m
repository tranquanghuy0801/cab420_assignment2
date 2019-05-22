%% Assignment 2 CAB420
% Group 22
% Student: Tran Quang Huy - n10069275
% Student: Nathan Armishaw - n9157191

% Clear workspace
clc; clear all; close all;
warning('off','all');

%% Support Vector Machines 

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

%Set constant values
C = 1000;
polyOrder = 2;
SD1 = 1;
SD2 = 1.5;

%% Part 1: 

% For the first three datasets, consider the linear, second 
% order polynomial, Gaussian of standard deviation 1 kernels 
% Use C = 1000 for consistency 

% Use linear kernels for dataset 1  
svm_test(@Klinear,[],C,set1_train,set1_test); 
% Use second order kernels for dataset 2 
svm_test(@Kpoly,polyOrder,C,set2_train,set2_test);  
% Use Gaussian of standard deviation 1 kernels for dataset 3  
svm_test(@Kgaussian,SD1,C,set3_train,set3_test);

%% Part 2 

TestError = zeros(3,1); % Initialize TestError variable 

% Train and test 4th dataset with a linear kernel
svm_linear4 = svm_train(set4_train,@Klinear,[],C); % Training 
y_linear4 = sign(svm_discrim_func(set4_test.X,svm_linear4)); % Prediction 
errors_linear = find(y_linear4 ~= set4_test.y); % Testing Error  
TestError(1) = length(errors_linear)/length(set4_test.y); % Output the result 
fprintf('Linear SVM: %g of 4th test examples  were misclassified.\n',...
    length(errors_linear)/length(set4_test.y));

% Train and test 4th dataset with a polynomial of degree 2 kernel 
svm_poly4 = svm_train(set4_train,@Klinear,polyOrder,C); % Training
y_poly4 = sign(svm_discrim_func(set4_test.X,svm_poly4)); % Prediction 
errors_poly = find(y_poly4 ~= set4_test.y); % Testing Error
TestError(2) = length(errors_poly)/length(set4_test.y);% Output the result 
fprintf('Polynomial SVM: %g of 4th test examples  were misclassified.\n',...
    length(errors_poly)/length(set4_test.y));

% Train and test 4th dataset with a Gaussian of standard deviation 1.5 kernels 
svm_gaussian4 = svm_train(set4_train,@Kgaussian,SD2,polyOrder); % Training 
y_gaussian4 = sign(svm_discrim_func(set4_test.X,svm_gaussian4)); % Prediction 
errors_gaussian = find(y_gaussian4 ~= set4_test.y);  % Testing Error  
TestError(3) =  length(errors_gaussian)/length(set4_test.y); % Output the result 
fprintf('Gaussian SVM: %g of 4th test examples  were misclassified.\n',...
    length(errors_gaussian)/length(set4_test.y));
Kernel = {'Linear';'Polynomial of degree 2';'Gaussian of std 1.5'}; 

% The test errors of 4th dataset trained on different kernels is as below: 
t = table(TestError,Kernel);
display(t);






