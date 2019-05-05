clc; clear all; close all;
warning('off','all'); 

%% PCA 
%
% Eigenfaces - Part A to E
%

% Load the data and display a few faces 
X = load('data/faces.txt'); 
img = reshape(X(2,:),[24 24]); 
imagesc(img); axis square; colormap gray; 

%% Part A: Subtract the mean of the face images to make the data zero-mean 
mean_X = mean(X); 
X0 = X - mean_X; 
[m,n] = size(X0); % The size of matrix data 
Sigma = 1/m .* X0' * X0; 
% Take the SVD of the data 
[U S V] = svd(Sigma); 
W = U * S; 

%% Part B: Compute the mean square error in SVD's approxination 
errors = zeros(1,10); % Store the MSE in SVD's approximation 
K = 1:10; 
for i = 1:length(K) 
    [U_k S_k V_k] = svds(Sigma,K(i)); 
    X0_svd = U_k * S_k * V_k'; 
    mse_svd = mean(mean((W * V' - X0_svd).^2)); 
    errors(i) = mse_svd;     
end 
figure(1); 
plot(K,errors); 
title('Plot the MSE in SVD approximation against the K'); 
xlabel('K'); ylabel('MSE'); 

%% Part C: Display a first few principal directions of the data 

alpha = 2 * median(abs(W(:,10))); 
direction1 = reshape(mean_X + alpha * V(:,10)',[24,24]); 
direction2 = reshape(mean_X - alpha * V(:,10)',[24,24]); 
figure(2); 
imagesc(direction1); axis square; colormap gray; 
figure(3);
imagesc(direction2); axis square; colormap gray; 

%% Part D: Latent Space methods 

idx = 15:25; % random indices of data 
figure(4); title('Latent Space methods');  hold on; axis ij; colormap(gray); 
range = max(W(idx,1:2)) - min(W(idx,1:2)); % find range of coordinates to be plotted 
scale = [200 200]./range;                  % want 24x24 to be visible 
for i=1:length(idx)
    imagesc(W(idx(i),1) * scale(1), W(idx(i),2) * scale(2), reshape(X(idx(i),:),24,24)); 
end 

%% Part E: Choose two faces and reconstruct using only K principal directions 

K_recover = [5,10,50]; % Use K principal directions 
% Image 1 with index 2 
img1 = X0(2,:); 
recovered_img1 = img1 * U(:,1:500) * U(:,1:500)'; 
figure(5); 
subplot(1,2,1); imagesc(reshape(X(2,:),24,24)); axis square; colormap gray; 
subplot(1,2,2); imagesc(reshape(recovered_img1,24,24)); axis square; colormap gray; 

%% Clustering 
%
% Part A to E
% 
%% Part A: Load the usual Iris data with 2 features and plot 
iris = load('data/iris.txt'); 
X_iris = iris(:,1:2); 
plot(X_iris(:,1),X_iris(:,2),'ro'); 
title('Plot of Iris data with 2 first features'); 

%% Part B: K-Means on the data 







