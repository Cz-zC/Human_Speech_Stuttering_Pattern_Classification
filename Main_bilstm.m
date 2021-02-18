%% New
outputs=5;
epochs=10;
global Acuracy xtest ytest con XT YT in layers options dly fact;
fact=10;
dly=0;
lenfeatures=13;
in=0;
Acuracy=zeros(10,1);
%% Loading Data:
filepath='C:\Users\HP\Documents\Stuttering Project\Solution\Labels\LibriStutter\Annotation\';'FileExtension';'.csv';
ds=tabularTextDatastore(filepath);
filepath='C:\Users\HP\Documents\Stuttering Project\Solution\Labels\LibriStutter\Audio\';'FileExtension';'.flac';
ads = audioDatastore(filepath);
%% Arranging Y data:
YTrain=cell(1,length(ds.Files));
for i=1:length(ds.Files)
    c=readtable(ds.Files{i});
    Current=[round((c.Var2).*fact),round((c.Var3).*fact),double(c.Var4)];
    k=zeros(Current(end,2)+1,2);
    k(:,1)=(0:Current(end,2));
    [m,n]=size(Current);
    for j=0:Current(end,2)
        for l=1:m
            if j>=Current(l,1) && j<=Current(l,2)
                k(j+1,2)=Current(l,3);
            end
        end
    end
    YTrain{i}=k(:,2);
end
fs=zeros(length(ds.Files),1);
xfs=cell(length(ds.Files),1);
 for i=1:length(ds.Files)
     [X,FS]=read(ads);
     fs(i)=FS.SampleRate;
     xfs{i,1}=X;
 end
 min_fs=min(fs);
 yfs=cell(length(ds.Files),1);
 for i=1:length(ds.Files)
    yfs{i} = resample(xfs{i},min_fs,fs(i));
 end
 %% Arranging X Data:
 XTrain=cell(1,length(ds.Files));
 Par=cell(1,length(ds.Files));
 for i=1:length(ds.Files)
    x=yfs{i};
    r=rem(length(x),length(YTrain{i}));
    %x=x(floor(r/2)+1:end-ceil(r/2));
    x=x(floor(r/2)+1:end-ceil(r/2));
    %x=x(1:end-r);
    q=(length(x)/length(YTrain{i}));
    P=reshape(x,[q,length(YTrain{i})]);
    Par{i}=P;
 end
%% Extract Features:
for i=1:length(Par)
    store=Par{i};
    [lenstore,n]=size(store);
    eval=cell(n,1);
    
%     aFE = audioFeatureExtractor('SampleRate',lenstore*10, ...
%     'mfcc');
    for j=1:n
        audioIn=store(:,j);
        win=rectwin(lenstore);
        S = stft(audioIn,"Window",win,"OverlapLength",round(lenstore/2));
        features = mfcc(S,min_fs,"LogEnergy","Replace");
        eval{j}=features';
    end
    XTrain{i}=eval;
end
%% Transposing to Training format:
count=0;
XTrain=XTrain';
YTrain=YTrain';
for i=1:length(YTrain)
    %XTrain{i}=rescale(XTrain{i});
    v=length(YTrain{i});
    count=count+v;
end
%% Data distributed to each sample:
TrainY=zeros(count,1);
TrainX=cell(count,1);
m=1;
for i=1:length(XTrain)
    V=YTrain{i};
    for j=1:length(XTrain{i})
        TrainX{m}=XTrain{i,1}{j,1};
        TrainY(m,1)=V(j,1);
        m=m+1;
    end
end

%% Delay Function:
ChangeX=TrainX;
ChangeY=TrainY;
TrainX= cell(length(ChangeY)-2*dly,1);
TrainY=zeros(length(ChangeY)-2*dly,1);
for i=dly+1:length(ChangeY)-dly
    TrainX{i-dly}=[ChangeX{i-(dly):i+dly}];
    TrainY(i-dly)=ChangeY(i);
end
%% Training Initialize:
TrainY=categorical(TrainY);
con=1;
[ytest,xtest]=Preparetest();
%% Layers:
layers=[ ...
        sequenceInputLayer(lenfeatures)
        bilstmLayer(100,'OutputMode','last')
        fullyConnectedLayer(outputs)
        softmaxLayer
        classificationLayer];
%% Setting the Network options:
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
%% Cross Validation:
error = crossval('mcr',TrainX,TrainY,'Predfun',@Trainingnetwork,'Stratify',TrainY);
%% Excel Storage
% b="mfcc";
% [mTr,nTr]=size(YT);
% [mTs,nTs]=size(ytest);
% [att,node]=size(XT{1});
% j=readtable('Datalog.xlsx','PreserveVariableNames',true);
% [m,n]=size(j);
% C=strcat('A',int2str(m+1),':AO',int2str(m+1));
% add=horzcat(m+1,mTr,mTs,att,node,epochs,(dly*2),Acuracy(in),b);
% add_len=size(add)
% xlswrite('Datalog.xlsx',add,1,C);
%% Training and Testing:
function [ypred] = Trainingnetwork(TrainX,TrainY,Xtest)
global xtest ytest con Acuracy XT YT in layers options;
[net,~] = trainNetwork(TrainX,TrainY,layers,options);
ypred = classify(net,Xtest, ...
    'MiniBatchSize',25, ...
    'SequenceLength','longest');
Ypred = classify(net,xtest, ...
    'MiniBatchSize',25, ...
    'SequenceLength','longest');
%% Fetch Best training set
Acuracy(con,1)=sum(Ypred == ytest)./numel(ytest);
disp(Acuracy(con,1));
[~,in]=max(Acuracy);
if in==con
    XT=TrainX;
    YT=TrainY;
end
con=con+1;  
end