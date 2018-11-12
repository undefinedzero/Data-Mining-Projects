clear all
tol_his = [];
for i = 1:3
    wid = 1;
    I = imread(['./I',num2str(i),'.bmp']);
    [row, col] = size(I);
    histo = myhist(I,wid);
    x=0:wid:255;
    tol_his = [tol_his, histo];
    figure,histogram(I,256);axis([0,255,0,1.1*max(histo)]);
    bar(x,histo);axis([0,255,0,1.1*max(histo)]);
%     saveas(gcf,['./histdet_I',num2str(i),'_',num2str(wid)],'png');
end