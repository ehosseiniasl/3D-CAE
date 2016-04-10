function X = max3d_3D(X, M)
    ll = size(X);
    B=X(M);
    B=B+rand(size(B))*1e-12;
    B=(B.*(B==repmat(max(B,[],1),[size(B,1) 1])));
%     b=B(B~=0);
%     b=b(1:size(B,2));
%     b=reshape(b,ll/2);
%     X = b;
    X(M) = B;
    reshape(X,ll);
end