function [E_proj, s, p] = mypca(img,dim,threshold)
    A = double(img);
    A_m = ones(size(A,1),1)*mean(A);
    B = A - A_m;
    
    C = B' * B/(size(B,2)-1);
    [E, D] = eig(C);
    [V,order] = sort(diag(D),'descend');
    E = E(:,order);
    
    p = 1;
    while(1)
        th = sum(V(1:p));
        if(th>=threshold*sum(V))
            E_proj = E(:, 1:p);
            th/sum(V)
            break;
        end
        p = p + 1;
    end
    
    U = B * E_proj;

    B_rec = U * E_proj';
    s = B_rec + A_m;
end