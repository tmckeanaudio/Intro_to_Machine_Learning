function LL = LogLikelihood(K,x)
%Function that computes the Negative-Log-Likelihood from the given Range
%Values, Prior parameters, and x/y-vectors that serve as inputs
w = 2*pi/K;
kx = zeros(1,length(K));
ky = zeros(1,length(K));
sig_x = 0.25;
sig_y = 0.25;
for i = 0:K-1
kx(1,i+1) = round(cos(w*i),6); ky(1,i+1) = round(sin(w*i),6);
end
base = [kx' ky'];
for i = 1:K
    r(i,:) = calculateRange(base,x,0.3);
end
ri = sum(r(:));
x_P = 0.2;
y_P = -0.1;
LL = -(ri+log(2*pi*sig_x*sig_y)+(1/2)*(x_P^2/sig_x^2 + y_P^2/sig_y^2));
end

