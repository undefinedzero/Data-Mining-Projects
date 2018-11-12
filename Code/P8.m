for i = 4:6
    for q = [0.1,1,10,100]
        f = imread(['./I',num2str(i),'.RGB.bmp']);
        [y, u, v] = Compress(f,q);
        g = Decompress(y,u,v,q);
        imshow(g);
        imwrite(g,['./dct_I',num2str(i),'_',num2str(q),'.png']);
    end
end