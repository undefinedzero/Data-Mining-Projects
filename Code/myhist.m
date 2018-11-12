function hist_out = myhist(image_in, width)

[row, col] = size(image_in);
divide = ceil(256.0 / width);
hist_out = zeros(divide,1);

for i = 1:row
    for j = 1:col
        index = ceil(double((image_in(i,j) + 1.0)) / width);
        hist_out(index) = hist_out(index) + 1;
    end
end