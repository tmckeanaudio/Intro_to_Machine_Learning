function [x,componentLabels] = generateDataFromGMM(N,gmmParameters,visualizationFlag)
% Generates N vector samples from the specified mixture of Gaussians
% Returns samples and their component labels
% Data dimensionality is determined by the size of mu/Sigma parameters
priors = gmmParameters.priors; % priors should be a row vector
meanVectors = gmmParameters.meanVectors
covMatrices = gmmParameters.covMatrices
n = size(gmmParameters.meanVectors,1) % Data dimensionality
C = length(priors); % Number of components
x = zeros(n,N); componentLabels = (rand(1,N) >= priors(1));
% Decide randomly which samples will come from each component
rndSample = rand(1,N); thresholds = [cumsum(priors),1];
for l = 0:C-1 % loop for class labels 0 and 1
    idx = find(componentLabels==l); % logical array that finds indices of classes 0 and 1
    if l == 0 % class 0
        for i = 1:2
        indl = find(rndSample <= thresholds(i)); 
        N0 = length(indl);
        componentLabels(1,indl) = l*ones(1,N0);
        rndSample(1,indl) = 1.1*ones(1,N0); % these samples should not be used again
        pdfParameters.Mean=meanVectors(:,i);
        pdfParameters.Cov=covMatrices(:,:,i);
        x(:,indl)= mvnrnd(pdfParameters.Mean',pdfParameters.Cov,N0)';
        end
    elseif l == 2 % Class 1
        N1 = length(idx);
        pdfParameters.Mean=meanVectors(:,l+1);
        pdfParameters.Cov=covMatrices(:,:,l+1);
        x(:,idx)= mvnrnd(pdfParameters.Mean',pdfParameters.Cov,N1)';
    end
end
    %pdfParameters.type = 'Gaussian';
    %x(:,indl) = generateRandomSamples(Nl,n,pdfParameters,0);
    % Matlab's built-in sampler for specified multivariate Gaussian pdf
    
if visualizationFlag==1 & 0<n & n<=3
figure
    if n==1
        plot(x,zeros(1,N),'.'); title('x~ 1D data Generated from Mixtures of Gaussians');
        xlabel('x-axis');
    elseif n==2
         plot(x(1,1:N),x(2,1:N),'.'); title('x~ 2D data Generated from Mixtures of Gaussians');
          xlabel('x-axis');ylabel('y-axis');
    elseif n==3
         plot3(x(1,:),x(2,:),x(3,:),'.'); title('x~ 3D data Generated from Mixtures of Gaussians');
         xlabel('x-axis');ylabel('y-axis'); zlabel('z-axis')
         
    end
end 
end 

