function [yorig,ys] = read_imgs(datadir,format)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明

if ~isempty(format)
    dirs = dir(fullfile(datadir,format));
else
    dirs = dir(fullfile(datadir));
end

fns = {dirs.name};
ys = {};
yorig = ys;
N=0;
for i = 1:length(fns)
    if isfile(strcat(datadir,fns{i}))
        N=N+1;
        yorig{N} = im2double(imread(strcat(datadir,fns{i})));
        ys{N} = strcat(datadir,fns{i});
    end
end
end

