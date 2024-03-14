function L=w2l(W, idx_closest)
% W2L weight matrix to Laplacian matrix
% 
% L=w2l(W)
% 
% by: KS Lu
% 20170712
%
sz_W = size(W);
C = zeros(sz_W);
if any(W<0)
    %error('W is not a valid weight matrix');
end

if (nargin > 1)
    C(idx_closest,idx_closest) = 2;
    L = C + diag(sum(W))-W+diag(diag(W));
else
    L = diag(sum(W))-W+diag(diag(W));
end


end