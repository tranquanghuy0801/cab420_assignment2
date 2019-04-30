clc; clear all; close all;
warning('off','all'); 

%% PCA && Clustering 

% Load the data and display a few faces 
X = load('data/faces.txt'); 
img = reshape(X(2,:),[24 24]); 
imagesc(img); axis square; colormap gray; 

%% Part A: Subtract the mean of the face images to make the data zero-mean 


