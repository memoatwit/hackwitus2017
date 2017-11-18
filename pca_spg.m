function [D, W_pca] = pca_spg(X)
% mu = mean(X);
[n,p]=size(X);
%Xm = X - repmat(mu, [n 1]);
Xm=X;
C = cov(Xm);
[W_pca,D] = eig(C);
[D, i] = sort(diag(abs(D)), 'descend');
W_pca = W_pca(:,i);

%variances per V
blah=cumsum(D) / sum(D);
disp(['First eigenvector covers ',num2str(blah(1)),'% of variance'])
minft = find( (cumsum(D) / sum(D)) > 0.9);
disp(['We need a min of ', num2str(minft(1)),' features to cover 90% of variance'])
disp(['First ', num2str(minft(1)),' feature(s) are:'])
i(1:minft(1))
end