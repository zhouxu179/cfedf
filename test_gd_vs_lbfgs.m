x = imread('rice.png');
x = imread('hestain.png');
% x = rgb2gray(x);
% % x = imresize(x,[32,32]);
% x = imread('coins.png');
x = imread('cameraman.tif');
% x = imread('lena256.png');

I = im2double(x);
eta = 1e-1;
sigma = 3;
nstd = 0.002;
hks = ceil(3*sigma);
hks0 = round(1.2*sigma);
ks = 2*hks0+1;
N = 3;
shape.name = 'gaussian';
% shape.name = 'pyramid';
% shape.name = 'ellipse';

shape.k = N+N;
gt = I(hks0+1:end-hks0,hks0+1:end-hks0);
[H,W] = size(gt);
depth = topography_sfk(gt,shape)*sigma;
depth_gt = depth;

gap = sigma/(N);
fs = gap:gap:sigma;


psfs = depth2psfs(depth,fs,eta,hks0);
ys = cell(N,1);
row = floor(sqrt(N));
col = ceil(N/row);
for n = 1:N
    ys{n} = svconv2(I,psfs{n}) + nstd*randn(H,W);
    subplot(row,col,n),imshow(ys{n}),title('blurred')
end
%% 
% figure,imshow(depth/max(depth(:))),title('true depth');
% hks = hks0 + 0;
% [xf,weights,map] = fusion(ys,2*hks,100);
% psnr_x = 10*log10(numel(gt)/norm(double(gt(:))-xf(:),'fro')^2);
% figure,imshow(xf),title(sprintf('fusion result,psnr=%.2f',psnr_x));

xf = I;
%% 
fs0 = ones(N-1,1)*sigma/2;
fs0 = fs(2:end)-fs(1);
fs0 = fs0(:);
% fs0 = fs_est;
% opts.depth = depth;
prior.name = 'lp';
prior.name = 'gaussian';
% prior.name = 'tv';
prior.p = 1;
adaptive_scale = 1;
lambda = 0.0001;
lambda_x = 0.0001;
alpha = 2/3;
lam_d = 1e-3;
N1 = 1;
N2 = 15;
lbfgs = 0;
irls.lambda =lambda;
irls.alpha = alpha;
irls.cost_display = 1;
irls.isnr_display = 0;
irls.groundtruth = [];
irls.out_iter = N1;
irls.inner_iter = N2;
opts = [];
opts.prior = prior;
opts.lambda_x = lambda_x;
opts.irls = irls;
opts.format = [];
opts.format = [];
opts.depth_iters = 100;
opts.lr_max = .1;
opts.lam_d = lam_d;
opts.adaptive_scale = adaptive_scale;
opts.blurs = ys;
opts.xk_iters = 1;
opts.lbfgs = lbfgs;
opts.tol = 1e-3;
tic
[depth,PSFs,deblur,fs_est] = bid([],hks,fs0,opts);
toc
dep = (depth - min(depth(:)))/(max(depth(:))- min(depth(:)));
imtool(dep)
%% 
% fs0 = ones(N-1,1)*sigma/2;
% fs0 = fs(2:end)-fs(1);
% fs0 = fs0(:);
% fs0 = fs_est;
% opts.depth = depth;

lbfgs = 1;
irls.lambda =lambda;
irls.alpha = alpha;
irls.cost_display = 1;
irls.isnr_display = 0;
irls.groundtruth = [];
irls.out_iter = N1;
irls.inner_iter = N2;
opts = [];
opts.prior = prior;
opts.lambda_x = lambda_x;
opts.irls = irls;
opts.format = [];
opts.format = [];
opts.depth_iters = 100;
opts.lr_max = .1;
opts.lam_d = lam_d;
opts.adaptive_scale = adaptive_scale;
opts.blurs = ys;
opts.xk_iters = 1;
opts.lbfgs = lbfgs;
tic
[depth1,PSFs1,deblur1,fs_est1] = bid([],hks,fs0,opts);
toc
dep1 = (depth1 - min(depth1(:)))/(max(depth1(:))- min(depth1(:)));
imtool(dep1);
%% 




yx = ys;
N = length(yx);
gt = I(hks+1:end-hks,hks+1:end-hks);
if size(deblur,1) > size(gt,1)
    deblur = deblur(hks+1:end-hks,hks+1:end-hks);
end

psnr = 10*log10(numel(gt)/norm(double(gt(:))-deblur(:),'fro')^2)
psnr1 = 10*log10(numel(gt)/norm(double(gt(:))-deblur1(:),'fro')^2)
err = norm(fs(2:end)' - fs(1) - fs_est,'fro')/norm(fs)
err1 = norm(fs(2:end)' - fs(1) - fs_est1,'fro')/norm(fs)

% if N<=4
%     xfstack = 0*xf;
% else
%     xfstack = fstack(ys,'nhsize',2*hks+1);
% end
% % xfstack = im2double(xfstack);
% psnr_fstack = 10*log10(numel(gt)/norm(double(gt(:))-xfstack(:),'fro')^2)
% if isempty(edf)
%     yx{N+2} = xfstack;
% else
%     yx{N+2} = edf;
% end
yx{N+1} = gt;
yx{N+2} = xf;
yx{N+3} = deblur1;
yx{N+4} = deblur;
X = cell2array(yx);
figure,montage(X);
title(sprintf('From 1 to %d:input; %d: groudtruth; %d: tv fusion, %.2fdB;%d: lbfgs, %.2fdB;  Last: gd, %.2fdB',N,N+1,N+2,psnr_x,N+3,psnr1,psnr));
%% 
