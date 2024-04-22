function [ GFT, Gfreq, Ahat ] = compute_GFT_noQ(Adj, A, idx_closest)

if(nargin > 2)
    L = w2l(Adj,idx_closest);
else
    L = w2l(Adj);
end
[GFT,D] = eig(L);       % GFT is right eigenvectors in columns D is diagonal matrix with eigenvalues
[Gfreq, idxSorted ] = sort( diag(D), 'ascend' );
GFT = GFT(:, idxSorted);
GFT(:,1)=abs(GFT(:,1));
GFT = GFT';
Gfreq(1) = abs(Gfreq(1));
Ahat = GFT * A;
end