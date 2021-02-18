fact=10;
win_length=1024;
 Loading Data:
filepath='C:\Users\reban\Documents\Speech Stuttering\Dataset\UCLASS\metadata\';'FileExtension';'.csv';
ds=tabularTextDatastore(filepath);
filepath='C:\Users\reban\Documents\Speech Stuttering\Dataset\UCLASS\audio\';'FileExtension';'.wav';
ads = audioDatastore(filepath);
% Arranging Y data:
fs=zeros(length(ds.Files),1);
xfs=cell(length(ds.Files),1);
yfs=cell(length(ds.Files),1);
for i=1:length(ds.Files)
    [X,FS]=audioread(ads.Files{i});
    fs(i)=FS;
    xfs{i}=X;
 end
 min_fs=min(fs);
 for i=1:length(ds.Files)
    [Numer, Denom] = rat(min_fs/fs(i));
    audiowrite('store.wav',(resample(xfs{i}, Numer, Denom)),min_fs);
    [yfs{i},f_s]=audioread('C:\Users\reban\Documents\Speech Stuttering\Transformer\store.wav','native');
    segments = strings(0);
    remain=ads.Files{i};
    while (remain ~= "")
        [token,remain]=strtok(remain,'\');
        segments=[segments;token];
    end
    %audiowrite(strcat('C:\Users\reban\Documents\Speech Stuttering\Transformer\Resampled\',segments(end)),yfs{i},min_fs);
 end
YTrain=cell(1,length(ds.Files));
sum=0;
g=cell(length(ds.Files),1); 
for i=1:length(ds.Files)
    g{i}=buffer(yfs{i},win_length,round(win_length/2));
    [m,n]=size(g{i});
    sum=sum+n;
    g{i}=g{i}';
end
h=zeros(sum,win_length);
o=1;
for i=1:length(ds.Files)
    k=g{i};
    [m,n]=size(g{i});
    h(o:o+m-1,:)=k;
    o=o+m;
end
Ytrain=cell(length(ds.Files),1);
G=cell(length(ds.Files),1);
 for i=1:length(ds.Files)
    c=readtable(ds.Files{i});
    Current=[round((c.Var3).*min_fs),round((c.Var4).*min_fs),double(c.Var8)];
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
    Ytrain{i}=k(:,2);
    [m_,n_]=size(g{i});
    index=round(linspace(1,length(Ytrain{i}),m_));
    J=zeros(length(index));
    for j=1:length(index)
        J=Ytrain{i}(index);
    end
    G{i}=J;
 end

H=zeros(sum,1);
O=1;
for i=1:length(ds.Files)
    k=G{i};
    [m,n]=size(G{i});
    H(O:O+m-1,:)=k;
    O=O+m;
end
Y=cell(sum,2);
con=0;
for i=1:length(H)
    con=con+1;
    Y{con,1}=strcat('Audio_',int2str(i),'.wav');
    Y{con,2}=H(i);
    %audiowrite(strcat('C:\Users\reban\Documents\Speech Stuttering\Transformer\Executable\','Audio_',int2str(i),'.wav'),h(i,:),min_fs);
end
%xlswrite('audio_label.xlsx',Y)
% %Extra
%writecell(Y,'audiolabel.csv')
%writematrix(h,'audiodata.csv')