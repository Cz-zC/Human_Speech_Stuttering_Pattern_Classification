filepath='C:\Users\HP\Documents\Stuttering Project\Solution\Labels\UCLASS\metadata\';'FileExtension';'.csv';
ds=tabularTextDatastore(filepath);
filepath='C:\Users\HP\Documents\Stuttering Project\Solution\Labels\UCLASS\audio\';'FileExtension';'.wav';
ads = audioDatastore(filepath);
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
    [yfs{i},f_s]=audioread('C:\Users\HP\Documents\Stuttering Project\Solution\Labels\Approach8\store.wav','native');
    segments = strings(0);
    remain=ads.Files{i};
    while (remain ~= "")
        [token,remain]=strtok(remain,'\');
        segments=[segments;token];
    end
    audiowrite(strcat('C:\Users\HP\Documents\Stuttering Project\Solution\Labels\Approach8\IN8_Resampled_2\',segments(end)),yfs{i},min_fs);
 end