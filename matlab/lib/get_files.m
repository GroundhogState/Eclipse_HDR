function fnames = get_files(dirname,extn)
% I do this often enough, this should just be a script....
%     Give me a directory and I'll give you the files with a given
%     extension
    if extn(1) == '.'
       extn = extn(2:end); 
    end

    d = dir(fullfile(dirname,sprintf('*.%s',extn)));
    d = {d.name};
    d = {d{~ismember(d,{'.' '..' })}};
    fnames = d;
    

end