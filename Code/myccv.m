function CCV = myccv(img,coherentPix, numberOfColors)
    if ~exist('coherentPix','var')
        coherentPix = 2;
    end
    if ~exist('numberOfColors','var')
        numberOfColors = 32;
    end
    
    CCV = zeros(2,numberOfColors);
    
%    Gaus = fspecial('gaussian',[5 5],2);
%    img = imfilter(img,Gaus,'same');
    
%    imgSize = (size(img,1)*size(img,2));
    thresh = coherentPix; %int32((coherentPrec/100) * imgSize);
    img = floor((img/(256/numberOfColors)));%bitshift(img,-3);
    
    for i=0:numberOfColors-1
        BW = (img==i);
        CC = bwconncomp(BW,8);
        compsSize = cellfun(@numel,CC.PixelIdxList);
        incoherent = sum(compsSize(compsSize>=thresh));
        CCV(:,i+1) = [incoherent; sum(compsSize) - incoherent];
    end
end

