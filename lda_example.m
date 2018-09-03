%Linear Discriminant Analysis on Iris Dataset

% X - Input data 150x4
% Y - Target data
clc;clear;close all;
load 'iris.mat'
X = X(1:2,1:100);
Y = Y(1:100);

% figure;
% plot(X(1,Y==1),X(2,Y==1),'*b');
% hold on
% plot(X(1,Y==2),X(2,Y==2),'sm');
% title('Original Data');

%% Creating trainig and test set with labels 

indx = randperm(size(X,2));
X = X(:,indx);
Y = Y(indx);

Per = 0.9; %Percentage of Training data 
Xtrain = X(:,1:ceil(Per*size(X,2)));
Ytrain = Y(1:ceil(Per*size(X,2)));

Xtest = X(:,1+ceil(Per*size(X,2)):end);
Ytest = Y(1+ceil(Per*size(X,2)):end);

[W,mu_trans] = lda_train(Xtrain,Ytrain);

%% Training Accuracy
label_tr = lda_test(Xtrain,W,mu_trans);

figure(1);
plot(Xtrain(1,Ytrain==1),Xtrain(2,Ytrain==1),'*r');
hold on
plot(Xtrain(1,Ytrain==2),Xtrain(2,Ytrain==2),'sm');
hold on
plot(Xtrain(1,label_tr==1),Xtrain(2,label_tr==1),'ok');
hold on
plot(Xtrain(1,label_tr==2),Xtrain(2,label_tr==2),'dg');

true = Ytrain;
correct=0;
for i=1:length(label_tr)
    if label_tr(i) == true(i)
        correct = correct+1;
    else 
        correct = correct-1;
    end
end
Train_Accuracy = (correct/length(true))*100

%% Test Accuracy 
        
label_te = lda_test(Xtest,W,mu_trans);
true = Ytest;

figure(2);
plot(Xtest(1,Ytest==1),Xtest(2,Ytest==1),'*r');
hold on
plot(Xtest(1,Ytest==2),Xtest(2,Ytest==2),'sm');
hold on
plot(Xtest(1,label_te==1),Xtest(2,label_te==1),'ok');
hold on
plot(Xtest(1,label_te==2),Xtest(2,label_te==2),'dg');

correct=0;
for i=1:length(label_te)
    if label_te(i) == true(i)
        correct = correct+1;
    else 
        correct = correct-1;
    end
end
Test_Accuracy = (correct/length(true))*100






