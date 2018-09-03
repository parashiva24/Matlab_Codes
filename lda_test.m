
function label = lda_test(X,W,mu)

[c,N] = size(X);
label = zeros(1,N);
d = zeros(c,N);
Xtrans = W'*X;
for i = 1:length(mu)
    d(i,:) = (Xtrans - mu(i)).^2;
end
    
for i=1:N
    if d(1,i)<d(2,i)
        label(i) = 1;
    elseif d(1,i)>d(2,i)
        label(i) = 2;
    else
        label(i) = NaN;
    end
end
end
