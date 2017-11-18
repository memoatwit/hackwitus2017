% Principal Component Analysis and Linear Discriminant Analysis 
%see:
% http://www.bytefish.de/blog/pca_lda_with_gnu_octave/
% and 
% http://research.cs.tamu.edu/prism/lectures/pr/pr_l10.pdf
%
%Parkinsons data example form
% https://archive.ics.uci.edu/ml/datasets/Parkinsons
clear all; close all;

%load data
data = dlmread('/Users/memo/Downloads/Win_MATLAB_scripts/wdbc/wpbc.data',',',0,1);
y = data(:,1);
X = [data(:,2:end)];

% 569 samples (instances) with 30 attributes from 2 classes: Benign or not
% n is num of samples
% p is number of features
% y is the class labels
%C is the number of classes
[n,p]=size(X);
size(y);
C = length(unique(y)); %number of classes
disp(["Number of instances: ",n]);
disp(["Number of features: ",p]);
disp(["Number of classes: ",C]);

%%
%PCA without normalization
[DPca, Wpca] = pca_spg(X);
Xm = X - repmat(mean(X), [n 1]);
Xproj = Xm*Wpca(:,1:C-1);

wine1 = Xproj((y==1),:);
wine2 = Xproj((y==0),:);


figure;
plot(wine1(:,:),'ro', 'markersize', 10, 'linewidth', 3); hold on;
plot(wine2(:,:),'gx', 'markersize', 10, 'linewidth', 3);
% plot(wine3(:,1), wine3(:,2),'bo', 'markersize', 10, 'linewidth', 3);
title('Wisconsin Diagnostic Breast Cancer: LDA (original data)')
legend('Benign','Malignant')

figure;
plot(wine1,0,'ro', 'markersize', 10, 'linewidth', 3); hold on
plot(wine2,0,'gx', 'markersize', 10, 'linewidth', 3);
title('Wisconsin Diagnostic Breast Cancer: PCA (original data)')
legend('Benign','Malignant'); legend BOXOFF

%%
%Linear Discriminant Analysis without normalization
[D, W_lda] = lda_spg(X,y);
Xm = X - repmat(mean(X), [n 1]);
Xproj = Xm * W_lda(:,1:C-1);

wine1 = Xproj((y==1),:);
wine2 = Xproj((y==0),:);
% wine3 = Xproj((y==3),:);

figure;
plot(wine1(:,:),'ro', 'markersize', 10, 'linewidth', 3); hold on;
plot(wine2(:,:),'gx', 'markersize', 10, 'linewidth', 3);
% plot(wine3(:,1), wine3(:,2),'bo', 'markersize', 10, 'linewidth', 3);
title('Wisconsin Diagnostic Breast Cancer: LDA (original data)')
legend('Benign','Malignant')

%%
%PCA (normalized)
% z-scores measure the distance of a data point from the mean in terms of the standard deviation. 
% This is also called standardization of data. 
% The standardized data set has mean 0 and standard deviation 1, 
% and retains the shape properties of the original data set (same skewness and kurtosis).
% You can use z-scores to put data on the same scale before 
% further analysis. 
% This lets you to compare two or more data sets with different units.

Xn = zscore_spg(X);
[DPca, Wpca] = pca_spg(Xn);
Xproj = Xn*Wpca(:,1:C-1);

wine1 = Xproj((y==1),:);
wine2 = Xproj((y==0),:);
% wine3 = Xproj((y==3),:);

figure;
plot(wine1(:,:),'ro', 'markersize', 10, 'linewidth', 3); hold on;
plot(wine2(:,:),'gx', 'markersize', 10, 'linewidth', 3);
% plot(wine3(:,1), wine3(:,2),'bo', 'markersize', 10, 'linewidth', 3);
title('Wisconsin Diagnostic Breast Cancer: PCA (normalized data)')
legend('Benign','Malignant')

minft = find( (cumsum(DPca) / sum(DPca)) > 0.9);
disp(['We need a min of ', num2str(minft(1)),' features to cover 90% of variance'])

%%
%Linear Discriminant Analysis with normalization
Xn = zscore_spg(X);
[D, W_lda] = lda_spg(Xn,y);
Xm = Xn - repmat(mean(Xn), [n 1]);
Xproj = Xm * W_lda(:,1:C-1);

wine1 = Xproj((y==1),:);
wine2 = Xproj((y==0),:);
% wine3 = Xproj((y==3),:);

% %seperate class 1
% dx=[-2.5, -0.5];
% dy=[-2, -0.1];
% p1=[-2.5, 2];
% p2=[-0.5, -0.1];
% fit1 = polyfit(dx,dy,1); 
% yfit1 = polyval(fit1,[-2.5:0.1:2.5]);


figure;
plot(wine1(:,:),'ro', 'markersize', 10, 'linewidth', 3); hold on;
plot(wine2(:,:),'gx', 'markersize', 10, 'linewidth', 3);
% plot(wine3(:,1), wine3(:,2),'bo', 'markersize', 10, 'linewidth', 3);
title('Wisconsin Diagnostic Breast Cancer: LDA (normalized data)')
legend('Benign','Malignant')
