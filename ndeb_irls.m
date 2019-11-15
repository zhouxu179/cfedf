function [x_fov, x, opt] = ndeb_irls(y,PSF,opt)
%fast IRLS for image deblurring with undetermined boundary conditions
% input kernel size must be odd
% This method is a faster version of the CG based method Zhou et al [1].
% It incorpates the techniques of mask decoupling [3] and FIRLS [2].
% References: 
% [1] X. Zhou, et al, A boundary condition based deconvolution framework for 
% image deblurring, J. Comput. Appl. Math. 261 (2014) 14¨C29.
% [2] X. Zhou, et al, Fast iteratively reweighted least squares for 
% lp regularized image deconvolution and reconstruction, IEEE ICIP, 2014,

% [3]  M. S. C. Almeida and M. A. T. Figueiredo; "Deconvolving images 
% with unknown boundaries using the alternating direction method of multipliers", 
% IEEE Transactions on Image Processing, vol. 22, No. 8, pp. 3074-3086, 2013.

% If you use this code to do NBID, please cite the above papers.
% Copyright (c) Xu Zhou <xuzhou@buaa.edu.cn>, Beihang University, 2015. 
%-------------------------------------------------------------------------
% Input: blurred image y, blur kernel h, and options opt
% Output: 
% x_fov : the deblurred image in the field of view, of the same size as y
% x: the full deblurred image.
% Please refer to [2] for more details on FIRLS and opt.

%------------------------------------------------------------------------- 

[M1, M2,m1, m2]=size(PSF);
hks1=floor(m1/2);
hks2=floor(m2/2);
x=padarray(y,[hks1,hks2],'replicate','both');


% first and second order dirivative filters
dxf=[0 0 0;0 1 -1;0 0 0];
dyf=[0 0 0;0 1 0;0 -1 0];
dyyf=[0 -1 0;0 2 0;0 -1 0];
dxxf=[0 0 0;-1 2 -1;0 0 0];
dxyf=[0 0 0;0 1 -1;0 -1 1];

dxfr=rot90(dxf,2);
dyfr=rot90(dyf,2);
dxxfr=rot90(dxxf,2);
dyyfr=rot90(dyyf,2);
dxyfr=rot90(dxyf,2);


%import parameters
lambda=opt.lambda;
w0=0.25;

% n_tol=opt.n_tol;
if isfield(opt,'alpha')
    alpha = opt.alpha ;
else
    alpha = 2/3;
end


if isfield(opt,'isnr_display')
else
    opt.isnr_display = 0;
end
if opt.isnr_display==1
    I=opt.groundtruth;
end
if isfield(opt,'cost_display')
else
    opt.cost_display = 0;
end
if isfield(opt,'inner_iter')
    N2=opt.inner_iter;
else
    N2=15;
end
if isfield(opt,'out_iter')
    N1=opt.out_iter;
else
    N1=5;
end
if isfield(opt,'epsilon')
    epsilon=opt.epsilon;
else
    epsilon=0.01;
end
c=alpha*lambda;
beta=alpha*lambda/epsilon^(2-alpha);

iter=0;
% Wx=ones(size(adx))*lambda;
% Wy=ones(size(ady))*lambda;
% Wxx=ones(size(adxx))*lambda*w0;
% Wyy=ones(size(adyy))*lambda*w0;
% Wxy=ones(size(adxy))*lambda*w0;
b = svconv2_conj(y,PSF);

while iter<N1
    iter=iter+1;
    %updates Ax and residual for cg
    dx=conv2(x,dxf,'valid');
    dy=conv2(x,dyf,'valid');
    dxx=conv2(x,dxxf,'valid');
    dyy=conv2(x,dyyf,'valid');
    dxy=conv2(x,dxyf,'valid');

    adx=abs(dx);
    ady=abs(dy); 
    adxx=abs(dxx);
    adyy=abs(dyy);
    adxy=abs(dxy);
    Wx=min(beta,c*adx.^(alpha-2));
    Wy=min(beta,c*ady.^(alpha-2));
    Wxx=min(beta,c*adxx.^(alpha-2))*w0;
    Wyy=min(beta,c*adyy.^(alpha-2))*w0;
    Wxy=min(beta,c*adxy.^(alpha-2))*w0;
    Ax = svconv2_conj(svconv2(x,PSF),PSF);
    if lambda>0
        Ax=Ax+conv2(Wx.*conv2(x,dxf,'valid'),dxfr);
        Ax=Ax+conv2(Wy.*conv2(x,dyf,'valid'),dyfr);
        Ax=Ax+conv2(Wxx.*conv2(x,dxxf,'valid'),dxxfr);
        Ax=Ax+conv2(Wyy.*conv2(x,dyyf,'valid'),dyyfr);
        Ax=Ax+conv2(Wxy.*conv2(x,dxyf,'valid'),dxyfr);
    end
    r = b - Ax;
    p = r;
    rsold = r(:)'*r(:);
    for i=1:N2        
        Ap=svconv2_conj(svconv2(p,PSF),PSF);
        if lambda>0
            Ap=Ap+conv2(Wx.*conv2(p,dxf,'valid'),dxfr);
            Ap=Ap+conv2(Wy.*conv2(p,dyf,'valid'),dyfr);
            Ap=Ap+conv2(Wxx.*conv2(p,dxxf,'valid'),dxxfr);
            Ap=Ap+conv2(Wyy.*conv2(p,dyyf,'valid'),dyyfr);
            Ap=Ap+conv2(Wxy.*conv2(p,dxyf,'valid'),dxyfr);
        end
        step_size = rsold / (p(:)'*Ap(:) );
        x = x + step_size*p;                    % update approximation vector
        r = r - step_size*Ap;                      % compute residual
        rsnew = r(:)'*r(:);
        p = r + (rsnew/rsold)*p;
        rsold = rsnew;
    end
         
    if opt.cost_display==1
        res = svconv2(x,PSF) - y;
        opt.cost1(iter)=0.5*norm(res,'fro')^2;
        opt.cost2(iter)=lambda*sum(sum(adx.^(alpha)+ady.^(alpha)+...
            w0*adxx.^(alpha)+w0*adyy.^(alpha)+w0*adxy.^(alpha)));
        opt.cost3(iter)=opt.cost1(iter)+opt.cost2(iter);
        fprintf('Outiter=%d,costf=%f,',iter,opt.cost3(iter));
    end
    if opt.isnr_display==1
        opt.isnr(iter)=20*log10(norm(y-I,'fro')/norm(x(hks1+1:end-hks1,hks2+1:end-hks2)-I,'fro'));
        fprintf('isnr=%f,beta=%f\n',opt.isnr(iter),beta);
    else
        fprintf('beta=%f\n',beta);
    end    
%     e(iter)=max(abs(x(:)-x_old(:)));
end
x_fov=x(hks1+1:end-hks1,hks2+1:end-hks2);
end



