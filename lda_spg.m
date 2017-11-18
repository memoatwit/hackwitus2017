function [D, W_lda] = lda_spg(X,y)
% LDA assumes that covar of classes equal, whereas Fisher does not.
%Both assume normal distribution for each probability
% maximizing the between-class scatter, while minimizing the within-class scatter at the same time.

%   dimension = columns(X);
[n,p]=size(X);
dimension = p; %number of avail feautures

labels = unique(y); %number of desired classes
C = length(labels);
%scatter within class
Sw = zeros(dimension,dimension);
%scatter between classes
Sb = zeros(dimension,dimension);
mu = mean(X);

for i = 1:C
    %grab all features within class
    Xi = X((y == labels(i)),:);
    [n,~] = size(Xi);
    mu_i = mean(Xi);
    XMi = bsxfun(@minus, Xi, mu_i);
    %scatter within class
    Sw = Sw + (XMi.' * XMi );
    %scatter between classes
    MiM =  mu_i - mu;
    Sb = Sb + n * (MiM.' * MiM);
end
%sort eigenvalues based on eigenvalues' confidence
% [W_lda, D] = eig(Sw\Sb);
% % [D2, i2] = sort(diag(abs(D)), 'descend');
% if matrices are ill-coniditioned:
[W_lda, D] = eig(pinv(Sw)*Sb);
[D, i] = sort(diag(abs(D)), 'descend');
W_lda = W_lda(:,i);

%hermissian might not come out real due to rounding.
%keep it real
W_lda=real(W_lda);

%variances per eigenvector, provides confidence per feature
blah=cumsum(D) / sum(D);
disp(['First eigenvector covers ',num2str(blah(1)),'% of variance'])
minft = find( (cumsum(D) / sum(D)) > 0.9);
disp(['We need a min of ', num2str(minft(1)),' features to cover 90% of variance'])
%list the features of interest
disp(['First ', num2str(minft(1)),' feature(s) are:'])
i(1:minft(1))
end