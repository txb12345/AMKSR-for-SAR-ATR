function [x] = KOMP_ONE_singleview(flag, zTY, YTY, A, tr_dimension, sparsity)

dimension = tr_dimension;
x = zeros(dimension, 1);
zk = zeros(dimension, 1);
At = [];
index = zeros(sparsity, 1);
for nIter = 1:sparsity
    proj = (zTY - zk' * YTY) * A; 
    [value, pos] = sort(abs(proj), 2, 'descend');
%     index(nIter) = pos(1);
    if flag == pos(1)
        index(nIter) = pos(2);
    else
        index(nIter) = pos(1);
    end
    At = [At, A(:, index(nIter))];
    K_s = At' * YTY * At;
    [vectors, values] = eig(K_s);
    if(size(find(values < 0), 1) ~= 0)
        for j = 1:size(values, 1)
            if(values(j, j) < 0)
                values(j, j) = 0;
            end
        end
        K_s = vectors * values * vectors';
    end
    xt = pinv(K_s) * (zTY * At)';
    zk = At * xt; 
    A(:, index(nIter)) = zeros(dimension, 1);
end
x(index(1:nIter)) = xt;


