% Image compression funtion
function [BY2,BU2,BV2] = Compress(orig_image,q)

RGB=orig_image;
%�����Ƕ�RGB�����������з��� 
R=RGB(:,:,1);  
G=RGB(:,:,2);  
B=RGB(:,:,3); 

%RGB->YUV  
Y=0.299*double(R)+0.587*double(G)+0.114*double(B); 
[xm, xn] = size(Y);
U=-0.169*double(R)-0.3316*double(G)+0.5*double(B);  
V=0.5*double(R)-0.4186*double(G)-0.0813*double(B);  

%����һ��8*8��DCT�任��֤     
T=dctmtx(8);
%����DCT�任 BY BU BV��double����  
BY=blkproc(Y,[8 8],'P1*x*P2',T,T');  
BU=blkproc(U,[8 8],'P1*x*P2',T,T');  
BV=blkproc(V,[8 8],'P1*x*P2',T,T');  
%��Ƶ����������   
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
%��Ƶ����������    
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
 %ʹ�������������������������
 BY2=blkproc(BY,[8 8],'round(x./P1)',a);  
 BU2=blkproc(BU,[8 8],'round(x./P1)',b);  
 BV2=blkproc(BV,[8 8],'round(x./P1)',b);  