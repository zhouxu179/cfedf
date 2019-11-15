 function [x,weights,map] = fusion(ys,hws,gamma)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
N = length(ys);
if isempty(hws)
    hws = 4;
end
if ~exist('gamma','var')
    gamma = 3;
end
ws = 2*hws+1;
kernel = ones(ws);

f1 = [0 0 0;0 1 -1;0 0 0];
f2 = [0 0 0;0 1 0;0 -1 0];


[H,W,C] = size(ys{1});
if C == 1
    weights = zeros(H,W,N);
    for i = 1:N
%         ypad = padarray(ys{i},[hws,hws],'replicate','both');
%         Avg = conv2(ypad,kernel,'valid');
%         Var = (ys{i} - Avg).^2;
%         weights(:,:,i) = Var;
        ypad = padarray(ys{i},[1,1],'replicate','both');
        dx = conv2(ypad,f1,'valid');
        dy = conv2(ypad,f2,'valid');
        grad = sqrt(dx.^2+dy.^2);
        grad_pad = padarray(grad,[hws,hws],'replicate','both');
        tv = conv2(grad_pad,kernel,'valid');
        weights(:,:,i) = tv;
    end
    
    %enhance weights
    [we_max,map] = max(weights,[],3);
    % avoid zero we_max
    we_max = max(we_max,1e-6);
    for i = 1:H
        for j = 1:W
            weights(i,j,map(i,j)) = we_max(i,j)*gamma;
        end
    end
            
    we_sum = sum(weights,3);
    x = 0;
    for i = 1:N
        weights(:,:,i) = weights(:,:,i)./we_sum;
        x = x + weights(:,:,i).*ys{i};
    end 
end
if C == 3
    yorig = ys;
    weights = zeros(H,W,N);
    for i = 1:N
        ys{i} = rgb2gray(ys{i});
%         ypad = padarray(ys{i},[hws,hws],'replicate','both');
%         Avg = conv2(ypad,kernel,'valid');
%         Var = (ys{i} - Avg).^2;
%         weights(:,:,i) = Var;
        ypad = padarray(ys{i},[1,1],'replicate','both');
        dx = conv2(ypad,f1,'valid');
        dy = conv2(ypad,f2,'valid');
        grad = sqrt(dx.^2+dy.^2);
        grad_pad = padarray(grad,[hws,hws],'replicate','both');
        tv = conv2(grad_pad,kernel,'valid');
        weights(:,:,i) = tv;
    end
    %enhance weights
    [we_max,map] = max(weights,[],3);
    % avoid zero we_max
    we_max = max(we_max,1e-6);
    for i = 1:H
        for j = 1:W
            weights(i,j,map(i,j)) = we_max(i,j)*gamma;
        end
    end    
    we_sum = sum(weights,3);
    x = zeros(H,W,C);
    for i = 1:N
        weights(:,:,i) = weights(:,:,i)./we_sum;
        for j = 1:C
            x(:,:,j) = x(:,:,j) + weights(:,:,i).*yorig{i}(:,:,j);
        end
    end     
end
    
    
end

