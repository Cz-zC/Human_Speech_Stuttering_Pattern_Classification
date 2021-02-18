%Initialize:
K=9;
Folds=10;
Acuracy=zeros(10,1);
miniBatchSize = 27;
outputs=8;
epochs=100;
con=1;
%Get data:
t=readtable('features_Libristutter.xlsx');
s=readtable('audio_label_Libristutter.xlsx');
t=table2array(t);
s=table2array(s);
[samples,features]=size(t);
samples=samples-1;
Xtrain=cell(samples,1);
Ytrain=s;  
for i=1:samples
    k=t(i+1,:)';
    Xtrain{i}=k;
end
Ytrain=categorical(Ytrain);
%Training parameters:
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
    'LearnRateDropFactor',0.1);
    %'Plots','training-progress');
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
end