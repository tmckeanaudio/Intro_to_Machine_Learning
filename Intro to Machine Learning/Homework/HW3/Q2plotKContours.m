function [Kx,Ky] = Q2plotKContours(K,xT,yT,P)
% Function to compute base x, base y of K landmarks and compute distance 
% between 2D base station coordinates and potential true position of
% vehicle using Prior knowledge and parameters of x_T and y_T
% Plot a Circle of Unit Radius that K landmarks reside on
viscircles([0 0],1,'Color','k','LineStyle','--','LineWidth',0.5); hold on
w = 2*pi/K;
Kx = zeros(1,length(K));
Ky = zeros(1,length(K));
for i = 0:K-1
Kx(1,i+1) = round(cos(w*i),6); Ky(1,i+1) = round(sin(w*i),6);
plot(Kx(1,i+1),Ky(1,i+1),'ro','LineWidth',2)
end
% Contour Plot for PDF of Vehicle Position
contour(xT,yT,P,10,'LineWidth',1.5)
xlabel('x'),ylabel('y'), grid on
title("\it{K} = "+K+" Landmarks on Circle of Unit Radius")
end

