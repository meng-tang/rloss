[imgsets gtsets] = textread('train_aug.txt','%s %s');
trainimgnames = cell(numel(imgsets), 1);
for i=1:numel(imgsets)
   imgname = imgsets{i};
   imgname = imgname(13:end-4);
   trainimgnames{i} = imgname;
end

fileID = fopen('train_id.txt','w');
for n=1:numel(trainimgnames)
    fprintf(fileID,'%s\n',trainimgnames{n});
end
fclose(fileID);

fileID = fopen('train.txt','w');
for n=1:numel(trainimgnames)
    fprintf(fileID,'JPEGImages/%s.jpg pascal_2012_scribble/%s.png\n',trainimgnames{n}, trainimgnames{n});
end
fclose(fileID);

[valimgnames] = textread('val_id.txt','%s');
fileID = fopen('val.txt','w');
for n=1:numel(valimgnames)
    fprintf(fileID,'JPEGImages/%s.jpg SegmentationClassAug/%s.png\n',valimgnames{n}, valimgnames{n});
end
fclose(fileID);