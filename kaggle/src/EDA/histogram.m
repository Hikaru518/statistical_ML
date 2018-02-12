echo off
clear all
home
%echo on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Kaggle EDA histogram                              %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Peiguang Wang   
% 2/11/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
%% Read in an image 
% 
display('Read image...')
  
im1 = imread('../../result/train_imgs/24.png');
im1 = double(im1)/255;

im2 = imread('../../result/train_imgs/13.png');
im2 = double(im2)/255;

im3 = imread('../../result/train_imgs/77.png');
im3 = double(im3)/255;

im4 = imread('../../result/train_imgs/97.png');
im4 = double(im4)/255;

%
%% conver into grayscale
I1 = rgb2gray(im1);
I2 = rgb2gray(im2);
I3 = rgb2gray(im3);
I4 = rgb2gray(im4);

%
%% normalization
I1 = (I1-min(I1(:)))/max(I1(:));
I2 = (I2-min(I2(:)))/max(I2(:));
I3 = (I3-min(I3(:)))/max(I3(:));
I4 = (I4-min(I4(:)))/max(I4(:));

%
%% display histogram
subplot(431)
imshow(im1);
title('original images');
subplot(432)
imshow(I1);
title('gray-scale images(after normalization)')
subplot(433)
imhist(I1);
title('histogram');

subplot(434)
imshow(im2);
subplot(435)
imshow(I2);
subplot(436)
imhist(I2);

subplot(437)
imshow(im3);
subplot(438)
imshow(I3);
subplot(439)
imhist(I3);

subplot(4,3,10)
imshow(im4);
subplot(4,3,11)
imshow(I4);
subplot(4,3,12)
imhist(I4);

%% Convertion to 1 type
figure(2)

subplot(431)
imshow(im1);
title('original images');
subplot(432)
imshow(I1);
title('after pixel convertion');
subplot(433)
imhist(I1);
title('histrogram (after convertion)');

subplot(434)
imshow(im2);
subplot(435)
imshow(1-I2);
subplot(436)
imhist(1-I2);

subplot(437)
imshow(im3);
subplot(438)
imshow(I3);
subplot(439)
imhist(I3);

subplot(4,3,10)
imshow(im4);
subplot(4,3,11)
imshow(I4);
subplot(4,3,12)
imhist(I4);
