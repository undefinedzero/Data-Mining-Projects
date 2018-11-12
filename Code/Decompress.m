
function reco_image = Decompress(orig_image_Y,orig_image_U,orig_image_V,q)
%解压缩
a=[
	16 11 10 16 24 40 51 61;  
	12 12 14 19 26 58 60 55;  
	14 13 16 24 40 57 69 55;  
	14 17 22 29 51 87 80 62;  
	18 22 37 56 68 109 103 77;  
	24 35 55 64 81 104 113 92;  
	49 64 78 87 103 121 120 101;  
	72 92 95 98 112 100 103 99;

];  
%高频分量量化表    
  b=[17 18 24 47 99 99 99 99;  
     18 21 26 66 99 99 99 99;  
     24 26 56 99 99 99 99 99;  
     47 66 99 99 99 99 99 99;  
     99 99 99 99 99 99 99 99;  
     99 99 99 99 99 99 99 99;  
     99 99 99 99 99 99 99 99;  
     99 99 99 99 99 99 99 99;]; 
 
a = q * a;
b = q * b;
YI=blkproc(orig_image_Y,[8 8],'round(x.*P1)',a);  
UI=blkproc(orig_image_U,[8 8],'round(x.*P1)',b);  
VI=blkproc(orig_image_V,[8 8],'round(x.*P1)',b);  

T=dctmtx(8);
YI = blkproc(YI, [8 8], 'P1*x*P2', T', T); 
UI = blkproc(UI, [8 8], 'P1*x*P2', T', T); 
VI = blkproc(VI, [8 8], 'P1*x*P2', T', T); 

%YUV转为RGB
RI=YI-0.001*UI+1.402*VI;  
GI=YI-0.344*UI-0.714*VI;  
BI=YI+1.772*UI+0.001*VI;

%经过DCT变换和量化后的YUV图像 
RGBI=cat(3,RI,GI,BI); 
RGBI=uint8(RGBI);  
reco_image = RGBI;
