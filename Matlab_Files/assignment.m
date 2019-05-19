%% Assignment 2 CAB420
% Group 22
% Student: Tran Quang Huy - n10069275
% Student: Nathan Armishaw - n9157191

% Clear workspace
clc; clear all; close all;
warning('off','all');

%% Section A: Support Vector Machines 
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

%% Question 1: 
% For the first three datasets, consider the linear, second 
% order polynomial, Gaussian of standard deviation 1 kernels 

%% Part 1: 

% Train first 3 datasets, plot the decision boundary and print the test
% errors 

% Plot the decision boundary and test errors with the linear model
svm_test(@Klinear,[],C,set1_train,set1_test); 
svm_test(@Klinear,[],C,set2_train,set2_test); 
svm_test(@Klinear,[],C,set3_train,set3_test); 

% Plot the decision boundary and test errors with the 2nd order 
% polynomial model
svm_test(@Kpoly,polyOrder,C,set1_train,set1_test); 
svm_test(@Kpoly,polyOrder,C,set2_train,set2_test); 
svm_test(@Kpoly,polyOrder,C,set3_train,set3_test); 

% Plot the decision boundary and test errors with the gaussian model of 
% standard deviation 1
svm_test(@Kgaussian,SD1,C,set1_train,set1_test); 
svm_test(@Kgaussian,SD1,C,set2_train,set2_test); 
svm_test(@Kgaussian,SD1,C,set3_train,set3_test); 

%% Part 2 
 
% Train 4th dataset with a linear, polynomial of degree 2 and Gaussian of
% standard deviation 1.5 kernels 
 
% Preallocate TestError and set Kernel
TestError = zeros(3,1);
Kernel = {'Linear';'Polynomial of degree 2';'Gaussian of std 1.5'}; 
 
% For the following chunks of code the following process was used: First the
% SVM model was trained on the training data, then the errors between the
% prediction and actual results in the test data was calculated. Then the
% fraction of errors was displayed.
 
% For linear model
svm_linear4 = svm_train(set4_train,@Klinear,[],C); 
y_linear4 = sign(svm_discrim_func(set4_test.X,svm_linear4));
errors_linear = find(y_linear4 ~= set4_test.y);
TestError(1) = length(errors_linear)/length(set4_test.y);
fprintf('Linear SVM: %g of 4th test examples  were misclassified.\n',...
    length(errors_linear)/length(set4_test.y));
 
% For polynomial model
svm_poly4 = svm_train(set4_train,@Kpoly,polyOrder,C); 
y_poly4 = sign(svm_discrim_func(set4_test.X,svm_poly4));
errors_poly = find(y_poly4 ~= set4_test.y);
TestError(2) = length(errors_poly)/length(set4_test.y);
fprintf('Polynomial SVM: %g of 4th test examples  were misclassified.\n',...
    length(errors_poly)/length(set4_test.y));
 
% For gaussian model
svm_gaussian4 = svm_train(set4_train,@Kgaussian,SD2,C); 
y_gaussian4 = sign(svm_discrim_func(set4_test.X,svm_gaussian4));
errors_gaussian = find(y_gaussian4 ~= set4_test.y);
TestError(3) =  length(errors_gaussian)/length(set4_test.y);
fprintf('Gaussian SVM: %g of 4th test examples  were misclassified.\n',...
    length(errors_gaussian)/length(set4_test.y));
 
% The test errors of 4th dataset trained on different kernels is as below: 
display(table(TestError,Kernel));


%% Section A: Bayes Classifiers
% Code is relevant to parts (A) and (B)
% Set the Training and Test data as well as other constants related to them
Xtr1 = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1];
Xtr2 = [0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1];
Ytr = [0,1,1,1,0,1,1,1,0,0,0,0,0,0,1,1];
Xtest = [0 1; 1 0; 1 1];
YtrLength = length(Ytr);
Ytr0Length = length(Ytr(Ytr==0));
Ytr1Length = length(Ytr(Ytr==1));

% Find the probabilities needed to create Joint/naive Bayes classifier
% Find out the percentage occurrence of each possible class(Ytr) value
% to do this divide number of occurrences by the length of the total array
% considered. Repeat this process to calculate all probabilities needed for 
% classification
P_0 = length(Ytr(Ytr==0)) / YtrLength;
P_1 = length(Ytr(Ytr==1)) / YtrLength;

% Find probabilities for Joint Bayes classifier, these are the percentage
% occurrence rates of an (X1,X2) combination for a specific y value
% P(x1,x2|y). The naming logic specifies the x1 value, x2 value then y
% value.
jbc_P_000 = length(Xtr1(Xtr1==0 & Xtr2==0 & Ytr == 0))/Ytr0Length;
jbc_P_010 = length(Xtr1(Xtr1==0 & Xtr2==1 & Ytr == 0))/Ytr0Length;
jbc_P_100 = length(Xtr1(Xtr1==1 & Xtr2==0 & Ytr == 0))/Ytr0Length;
jbc_P_110 = length(Xtr1(Xtr1==1 & Xtr2==1 & Ytr == 0))/Ytr0Length;
jbc_P_001 = length(Xtr1(Xtr1==0 & Xtr2==0 & Ytr == 1))/Ytr1Length;
jbc_P_011 = length(Xtr1(Xtr1==0 & Xtr2==1 & Ytr == 1))/Ytr1Length;
jbc_P_101 = length(Xtr1(Xtr1==1 & Xtr2==0 & Ytr == 1))/Ytr1Length;
jbc_P_111 = length(Xtr1(Xtr1==1 & Xtr2==1 & Ytr == 1))/Ytr1Length;

% Find probabilities for naive Bayes classifier, these are the percentage
% occurrence rates of P(x1|y) and P(x2|y). The naming logic specifies the 
% x value then y value.
nbc_x1_P_00 = length(Xtr1(Xtr1==0 & Ytr == 0))/Ytr0Length;
nbc_x1_P_10 = length(Xtr1(Xtr1==1 & Ytr == 0))/Ytr0Length;
nbc_x2_P_00 = length(Xtr1(Xtr2==0 & Ytr == 0))/Ytr0Length;
nbc_x2_P_10 = length(Xtr1(Xtr2==1 & Ytr == 0))/Ytr0Length;
nbc_x1_P_01 = length(Xtr1(Xtr1==0 & Ytr == 1))/Ytr1Length;
nbc_x1_P_11 = length(Xtr1(Xtr1==1 & Ytr == 1))/Ytr1Length;
nbc_x2_P_01 = length(Xtr1(Xtr2==0 & Ytr == 1))/Ytr1Length;
nbc_x2_P_11 = length(Xtr1(Xtr2==1 & Ytr == 1))/Ytr1Length;


%% Section B: PCA 
% Eigenfaces - Part A to E

% Load the data and display a few faces 
X = load('data/faces.txt'); 
img = reshape(X(2,:),[24 24]); 
figure
imagesc(img); axis square; colormap gray; 

%% Part A: Subtract the mean of the face images to make the data zero-mean 
mean_X = mean(X); 
X0 = X - mean_X; 
% Take the SVD of the data 
[U, S, V] = svd(X0);
W = U * S; 

%% Part B: Compute the mean square error in SVD's approximation 
errors = zeros(1,10); % Store the MSE in SVD's approximation  
K = 1:10; 
for i = 1:length(K) 
    [U_k, S_k, V_k] = svds(X0,K(i)); 
    X0_svd = U_k * S_k * V_k'; 
    mse_svd = mean(mean((X0 - X0_svd).^2)); 
    errors(i) = mse_svd;     
end 
figure 
plot(K,errors); 
title('Plot the MSE in SVD approximation against the K'); 
xlabel('K'); ylabel('MSE'); 

%% Part C: Display a first few principal directions of the data 

alpha = 2 * median(abs(W(:,10))); 
direction1 = reshape(mean_X + alpha * V(:,10)',[24,24]); 
direction2 = reshape(mean_X - alpha * V(:,10)',[24,24]); 
figure 
imagesc(direction1); axis square; colormap gray; 
title('Direction +')
figure
imagesc(direction2); axis square; colormap gray; 
title('Direction -')


%% Part D: Latent Space methods 

idx = 15:25; % random indices of data 
figure
title('Latent Space methods');  hold on; axis ij; colormap(gray); 
range = max(W(idx,1:2)) - min(W(idx,1:2)); % find range of coordinates to be plotted 
scale = [200 200]./range;                  % want 24x24 to be visible 
for i=1:length(idx)
    imagesc(W(idx(i),1) * scale(1), W(idx(i),2) * scale(2), reshape(X(idx(i),:),24,24)); 
end 

%% Part E: Choose two faces and reconstruct using only K principal directions 

K_recover = [5,10,50]; % Use K principal directions 
% Choose random two faces 
indices = randperm(size(X0,1));
index1 = indices(1); index2 = indices(2); 
img1 = X(index1,:);
img2 =  X(index2,:);
figure 
title('Construct image using K principal directions'); 

% Iterate over the 3 k values then reconstruct and plot the image using 
% that k value, as well as the original image
for i = 1:length(K_recover)
    [U1 S1 V1] = svds(img1,K_recover(i)); 
    recovered_img1 = U1 * S1 * V1';
    subplot(2,4,i); 
    imagesc(reshape(recovered_img1,24,24)); axis square; colormap gray; 
    title([num2str(K_recover(i))]);
    [U2 S2 V2] = svds(img2); 
    recovered_img2 = U2 * S2 * V2';
    subplot(2,4,i+4); 
    imagesc(reshape(recovered_img2,24,24)); axis square; colormap gray; 
    title([num2str(K_recover(i))]);
end 
subplot(2,4,4); imagesc(reshape(img1,24,24)); axis square; colormap gray; 
title('Original Image'); 
subplot(2,4,8); imagesc(reshape(img2,24,24)); axis square; colormap gray; 
title('Original Image'); 

%% Section B: Clustering 
%
% Part A to E
% 
%% Part A: Load the usual Iris data with 2 features and plot 
iris = load('data/iris.txt'); 
X_iris = iris(:,1:2); 
figure 
plot(X_iris(:,1),X_iris(:,2),'ro'); 
title('Plot of Iris data with 2 first features'); 

%% Part B: K-Means on the data 

% Run k-means with k = 5 
figure
title('k-means on the data with k = 5'); 
[z5 c5 sumd5] = kmeans(X_iris,5,'farthest',100); 
plotClassify2D([],X_iris,z5); 
hold on 
plot(c5(:,1),c5(:,2),'kx'); 

% Run k-means with k = 20 
figure
title('k-means on the data with k = 20'); 
[z20 c20 sumd20] = kmeans(X_iris,20,'random',100); 
plotClassify2D([],X_iris,z20); 
hold on 
plot(c20(:,1),c20(:,2),'kx'); 

%% Part C: Run agglomerative clustering on the data 

% Using single linkage with k = 5 and k = 20 
[z5_min_agg join] = agglomCluster(X_iris,5,'min'); 
[z20_min_agg join] = agglomCluster(X_iris,20,'min'); 
figure
subplot(1,2,1); 
plotClassify2D([],X_iris,z5_min_agg); 
title('K = 5'); 
subplot(1,2,2); 
plotClassify2D([],X_iris,z20_min_agg); 
title('K = 20');

% Using complete linkage with k = 5 and k = 20 
[z5_max_agg join] = agglomCluster(X_iris,5,'max'); 
[z20_max_agg join] = agglomCluster(X_iris,20,'max'); 
figure
subplot(1,2,1); 
plotClassify2D([],X_iris,z5_max_agg); 
title('K = 5'); 
subplot(1,2,2); 
plotClassify2D([],X_iris,z20_max_agg); 
title('K = 20');


%% Part D: Run the EM Gaussian Mixture Model 
% Use doPlot = true in emCluster to observe the evolution of mixture 
% components’ locations and shapes 

% EM Gaussian mixture model with k = 5
[z5_em,T,soft,ll] = emCluster(X_iris,5,'farthest',10); 

% EM Gaussian mixture model with k = 20 
[z20_em,T,soft,ll] = emCluster(X_iris,20,'farthest',10); 
