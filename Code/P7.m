clear all
for i = 1:3
    cut = 0.8;
    I = imread(['./I',num2str(i),'.bmp']);
    I_re = double(I);
    [M, N] = size(I);
    [E, s, p] = mypca(I,M,cut);
    I_re = s;
    figure, imshow(I);
    figure, imshow(I_re/255);
    imwrite(I_re/255,['./pca_I',num2str(i),'_',num2str(cut),'.png']);
end