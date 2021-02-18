function[Trainy,Trainx]=Preparetest()
global dly fact;
% Loading Data:
filepath='C:\Users\HP\Documents\Stuttering Project\Solution\Labels\Approach8\LibriStutter\LibriStutter Part 3\Annotation Libri 3\';'FileExtension';'.csv';
ds=tabularTextDatastore(filepath);
filepath='C:\Users\HP\Documents\Stuttering Project\Solution\Labels\Approach8\LibriStutter\LibriStutter Part 3\Audio Libri 3\';'FileExtension';'.flac';
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
Trainy=zeros(count,1);
Trainx=cell(count,1);
m=1;
for i=1:length(XTrain)
    V=YTrain{i};
    for j=1:length(XTrain{i})
        Trainx{m}=XTrain{i,1}{j,1};
        Trainy(m,1)=V(j,1);
        m=m+1;
    end
end

%% Delay Function:
ChangeX=Trainx;
ChangeY=Trainy;
Trainx= cell(length(ChangeY)-2*dly,1);
Trainy=zeros(length(ChangeY)-2*dly,1);
for i=dly+1:length(ChangeY)-dly
    Trainx{i-dly}=[ChangeX{i-(dly):i+dly}];
    Trainy(i-dly)=ChangeY(i);
end
%% Training Initialize:
Trainy=categorical(Trainy);
end