clear all
err = [];
for s = 0:32:8192
    %get ccv
    ccv = [];
    for i = [1,3]
        I = imread(['./I',num2str(i),'.bmp']);
        [row, col] = size(I);
        ccv_fig = myccv(I,s,32);
        ccv = [ccv;ccv_fig];
%         figure, bar(ccv_fig(1,:));
%         saveas(gcf,['./ccv_b_I',num2str(i),'_',num2str(s)],'png');
%         figure, bar(ccv_fig(2,:));
%         saveas(gcf,['./ccv_a_I',num2str(i),'_',num2str(s)],'png');
    end
    %test diff
    err_ccv = sum(abs(ccv(1,:)-ccv(3,:)) + abs((ccv(2,:)-ccv(4,:))));
    err_his = sum(abs(ccv(1,:)+ccv(2,:) - (ccv(3,:)+ccv(4,:))));
    err = [err,[err_ccv;err_his]];
end
s = 0:32:8192
plot(s,err(1,:),'Linewidth',3);hold on;
plot(s,err(2,:),'Linewidth',3);