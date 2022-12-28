function ri = calculateRange(xiyi,xTyT,sigma)
% Calculate the Range Measurements provided from 2D coordinates which is
% the Euclidean Distance with additional zero-mean Gaussian Noise
[m,~] = size(xiyi);
di = pdist2(xiyi,xTyT,'euclidean');
while(1)
ni = sigma*randn(m,1); % zero-mean Gaussian noise with std = sigma
ri = di ;%+ ni;
if ri > 0
    break % only exit loop if range measurement is nonnegative
end
end
ri = ri'; % output range measurement as row vector
end

