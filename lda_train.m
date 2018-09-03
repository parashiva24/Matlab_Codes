

function [W,mu_trans] = lda_train(X,Y)

K = unique(Y) ; %class labels
Mu = mean(X,2);
N = size(X,2);
Sw = 0;
Sb = 0;
for j=1:length(K)
    xj = X(:,Y==j);
    muj = repmat(mean(xj,2),1,length(xj));   
    Sw = Sw + (length(xj)/N) * (xj-muj)*(xj-muj)' ;
    Sb = Sb + (muj(:,1)-Mu)*(muj(:,1)-Mu)' ;
end

[v,lambda] = eig(Sw\Sb) ; % Eigen vectors of inv(Sw)*Sb;
lambda = diag(lambda);
for i=1:length(lambda)
    if lambda(i)<=1e-3
        break;
    end
    W(:,i) = v(:,i);
end

Xtrans = W'*X;

for i=1:length(K)
    mu_trans(i) = mean(Xtrans(:,Y==i)); %mean of the transformed class
end

end

