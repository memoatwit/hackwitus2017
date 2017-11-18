function z = zscore_spg(x)
%2 Dim array zscore
% % z-scores measure the distance of a data point from the mean in terms of the standard deviation.
% This is also called standardization of data.
% The standardized data set has mean 0 and standard deviation 1,
% and retains the shape properties of the original data set (same skewness and kurtosis).
% % You can use z-scores to put data on the same scale before further analysis.
% This lets you to compare two or more data sets with different units.

% [n,p]=size(X);
% Xm = X - repmat(mean(X), [n p]);
% z = Xm/std(X);

flag = 0;
% Figure out which dimension to work along.
dim = find(size(x) ~= 1, 1);
if isempty(dim), dim = 1; end

% Compute X's mean and sd, and standardize it
mu = mean(x,dim);
sigma = std(x,flag,dim);
sigma0 = sigma;
sigma0(sigma0==0) = 1;
z = bsxfun(@minus,x, mu);
z = bsxfun(@rdivide, z, sigma0);
end
