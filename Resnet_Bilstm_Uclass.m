%Initialize:
K=9;
Folds=10;
Acuracy=zeros(10,1);
miniBatchSize = 27;
outputs=8;
epochs=20;
con=1;
%Get data:
t=readtable('SMOTE_features_Uclass.xlsx');
s=readtable('SMOTE_labels_Uclass.xlsx');
t=table2array(t);
s=table2array(s);
t=t(1:end-6,:);
s=s(1:end-6,:);

%XT=cell(150,1);
% T=t(602:751,:);
% S=s(601:750);
%S=categorical(S);
%for i=1:150
%    XT{i}=T(i,:)';
%end
%t=t(1:601,:);
%s=s(1:600);
[samples,features]=size(t);
perbatch=floor(samples/outputs);
tt=reshape(t,[samples*features,1]);
idx=cat[1,t(1:)]
Xtrain=cell(samples,1);
Ytrain=s;  
for i=1:samples
    k=t(i,:)';
    Xtrain{i}=k;
end
 Ytrain=categorical(Ytrain);
% Training parameters:
layers=[ ...
        sequenceInputLayer(features)
        bilstmLayer(100,'OutputMode','last')
        fullyConnectedLayer(outputs)
        softmaxLayer
        classificationLayer];
% Setting the Network options:
options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',epochs, ...
    'MiniBatchSize',25, ...
    'GradientThreshold',1, ...
    'SequenceLength','longest', ...
    'shuffle','never',...
    'Verbose',1,...
    'InitialLearnRate', 1e-5,...
    'LearnRateDropFactor',0.1,...
    'Plots','training-progress');
foldlen=samples/Folds;
%Cross Validation:
acc=zeros(Folds,1);
for i=1:Folds
    TestX=Xtrain(foldlen*(i-1)+1:foldlen*i);
    TestY=Ytrain(foldlen*(i-1)+1:foldlen*i);
    testfold=foldlen*(i-1)+1:foldlen*i;
    trainfold=1:samples;
    for j=1:foldlen
        trainfold=trainfold(trainfold~=testfold(j));
    end
    TrainX=Xtrain(trainfold);
    TrainY=Ytrain(trainfold);
    net=trainNetwork(TrainX,TrainY,layers,options);
    YPred = classify(net,TestX, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest');
    acc(i) = sum(YPred == TestY)./numel(TestY);
    plotconfusion(TestY,YPred)
    disp(acc(i));
end