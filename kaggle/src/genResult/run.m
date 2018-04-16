path = './stage2_test_final';
d = dir(path);
dirList = d;

for i = 1:length(d)
    if(isequal(dirList(i).name,'.')||... % ignore . and ..
       isequal(dirList(i).name,'..'))                % ignore files
           continue;
    end
    
    number = floor(i/50);
    dirName = ['stage2_test_divide/', int2str(number)]; 
    if ~exist(dirName,'dir')
        mkdir(dirName);
    end
    
    newDir = [dirName '/',dirList(i).name];
    if ~exist(newDir,'dir')
        mkdir(newDir);
    end
    
    srcDir = ['./stage2_test_final/', dirList(i).name];
    desDir = newDir;
    
    copyfile(srcDir,desDir); 
end