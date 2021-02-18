filepath='C:\Users\reban\Documents\Speech Stuttering\Dataset\UCLASS\metadata\';'FileExtension';'.csv';
ds=tabularTextDatastore(filepath);
filepath='C:\Users\reban\Documents\Speech Stuttering\Dataset\UCLASS\audio\';'FileExtension';'.wav';
ads=audioDatastore(filepath);
seconds=1;
window_length=25/1000;
overlap_length=10/1000;
% Arranging Y data:
fs=zeros(length(ds.Files),1);
xfs=cell(length(ds.Files),1);
yfs=cell(length(ds.Files),1);
Yfs=cell(length(ds.Files),1);
for i=1:length(ds.Files)
    [X,FS]=audioread(ads.Files{i});
    fs(i)=FS;
    xfs{i}=X;
end
min_fs=min(fs);
for i=1:length(ds.Files)
    [Numer, Denom] = rat(min_fs/fs(i));
    audiowrite('store.wav',(resample(xfs{i}, min_fs, fs(i))),min_fs);
    [yfs{i},f_s]=audioread('C:\Users\reban\Documents\Speech Stuttering\Transformer\store.wav','native');
end
%Offset calculation
offset=zeros(length(ds.Files),1);
windows=zeros(length(ds.Files),1);
for i=1:length(ds.Files)
    windows(i)=floor(length(yfs{i})/(min_fs*seconds));
    including_offset=windows(i)*min_fs*seconds;
    offset(i)=floor((length(yfs{i})-including_offset)/2);
end
zfs=cell(sum(windows(1:end)),1);
win_len=min_fs*seconds;
k=1;
for i=1:length(ds.Files)
    Yfs{i}=yfs{i}(offset(i)+1:end-offset(i));
    for j=1:windows
        zfs{k}=Yfs{i}((j-1)*win_len+1:(j)*win_len);
        k=k+1;
    end
end
spec_num=k-1;
digits=numel(num2str(spec_num));
for i=1:spec_num
    spectrogram(double(zfs{i}),round(window_length*win_len/seconds),round(overlap_length*win_len/seconds),'yaxis');
    H = getframe(gca);
    formatspec=strcat('%0',int2str(digits),'d');
    filename=strcat('Executable_Uclass\Spectrogram_',num2str(i,formatspec),'.jpg');
    imwrite(H.cdata, filename);
    disp(i);
end
q=zeros(length(ds.Files),win_len);
for i=1:length(ds.Files)
    q(i,:)=zfs{i};
end
disp(std(double(q),0,2));

labelperfile=spec_num/length(ds.Files);
ylabel=zeros(labelperfile,length(ds.Files));
 for i=1:length(ds.Files)
    c=readtable(ds.Files{i});
    Current=[round((c.Var3).*min_fs),round((c.Var4).*min_fs),double(c.Var8)];
    [stamps,variables]=size(Current);
    start=Current(1,1);
    finish=Current(end,2);
    duration=finish-start;
    remain=floor(rem(duration,labelperfile));
    start=start+floor(remain/2);
    finish=finish-floor(remain/2);
    duration=finish-start;
    in_store=zeros(labelperfile,1);
    timestamps=round(linspace(0,duration,labelperfile));
    for j=1:labelperfile
        for k=1:stamps
            if timestamps(j)>=Current(k,1)&& timestamps(j)<=Current(k,2)
                in_store(j)=Current(k,3);
            end
        end
    end
    ylabel(:,i)=in_store;
 end
 labels=reshape(ylabel,spec_num,1);
 xlswrite('audio_label_Uclass.xlsx',labels);