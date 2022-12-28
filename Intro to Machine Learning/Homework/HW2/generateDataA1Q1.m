function [x,labels] = generateDataA1Q1(N)
%N = 100;
figure(1), clf,     %colors = 'bm'; markers = 'o+';
classPriors = [0.65,0.35];
labels = (rand(1,N) >= classPriors(1)); % Create random-valued vector from 0 to 1 and check whether values are >= 0.65
for l = 0:1 % For loop for class labels 0 and 1
    indl = find(labels==l); % logical array that finds indices of classes 0 and 1
    if l == 0 % class 0
        N0 = length(indl);
        w0 = [0.5,0.5]; mu0 = [3 0;0 3];
        Sigma0(:,:,1) = [2 0;0 1]; Sigma0(:,:,2) = [1 0;0 2];
        gmmParameters.priors = w0; % priors should be a row vector
        gmmParameters.meanVectors = mu0;
        gmmParameters.covMatrices = Sigma0;
        [x(:,indl),components] = generateDataFromGMM(N0,gmmParameters);
        plot(x(1,indl(components==1)),x(2,indl(components==1)),'mo'), hold on, 
        plot(x(1,indl(components==2)),x(2,indl(components==2)),'go'), hold on, 
        plot(mu0,'k+'), hold on,
        
    elseif l == 1 % class 1
        m1 = [2;2]; C1 = eye(2);
        N1 = length(indl);
        x(:,indl) = mvnrnd(m1,C1,N1)';
        plot(x(1,indl),x(2,indl),'b+'), hold on,
        axis equal,
    end
end
%%%
function [x,labels] = generateDataFromGMM(N,gmmParameters)
% Generates N vector samples from the specified mixture of Gaussians
% Returns samples and their component labels
% Data dimensionality is determined by the size of mu/Sigma parameters
priors = gmmParameters.priors; % priors should be a row vector
meanVectors = gmmParameters.meanVectors;
covMatrices = gmmParameters.covMatrices;
n = size(gmmParameters.meanVectors,1); % Data dimensionality
C = length(priors); % Number of components
x = zeros(n,N); labels = zeros(1,N); 
% Decide randomly which samples will come from each component
u = rand(1,N); thresholds = [cumsum(priors),1];
for l = 1:C
    indl = find(u <= thresholds(l)); Nl = length(indl);
    labels(1,indl) = l*ones(1,Nl);
    u(1,indl) = 1.1*ones(1,Nl); % these samples should not be used again
    x(:,indl) = mvnrnd(meanVectors(:,l),covMatrices(:,:,l),Nl)';
end