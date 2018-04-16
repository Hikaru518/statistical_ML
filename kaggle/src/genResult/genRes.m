wf = fopen('res.csv','w+');
fprintf(wf,'ImageId');
fprintf(wf,',');
fprintf(wf,'EncodedPixels');
fprintf(wf,'\r\n');
fclose(wf);

%
%% read
for jj = 0:60
    fid = fopen([int2str(jj) '.csv']);
    C = textscan(fid,'%s %s','Delimiter',',');
    fclose(fid);
    
    wf = fopen('res.csv','a+');
    for ii = 2:length(C{1});
        xx = C{1}(ii);
        fprintf(wf,xx{1});
        fprintf(wf,',');
        yy = C{2}(ii);
        fprintf(wf,yy{1});
        fprintf(wf,'\r\n');
    end
    fclose(wf);
    
    disp(jj);
end

%%
% wf = fopen('res.csv','a+');
% for ii = 2:length(C{1});
%     xx = C{1}(ii);
%     fprintf(wf,xx{1});
%     fprintf(wf,',');
%     yy = C{2}(ii);
%     fprintf(wf,yy{1});
%     fprintf(wf,'\r\n');
% end
% fclose(wf);

% 
% xx = C{1}(2);
% fprintf(wf,xx{1});
% fprintf(wf,',');
% yy = C{2}(2);
% fprintf(wf,yy{1});
% fprintf(wf,'\r\n');
% xx = C{1}(2);
% fprintf(wf,xx{1});
% fprintf(wf,',');
% yy = C{2}(2);
% fprintf(wf,yy{1});
% fprintf(wf,'\r\n');
% fclose(wf);

