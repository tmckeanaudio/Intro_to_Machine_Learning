function [data,classLabels] = HW2gmmData2D(N,Parameters)
% N = # of samples to generate
% Returns data and class labels
% Determine dimensions of data from parameters
%
rng(7);
% Gaussian Mixture Model Specifications
classPriors = Parameters.priors; % Class Priors
numClasses = length(classPriors);     % # of classes
component4label = Parameters.component4label; % Vector stating # of components for each class
% Count how many components for each class there are
for i = 0:numClasses-1
    numComponent4eachClass(i+1) = sum(component4label == i);
end
componentWeights = Parameters.componentWeights; % Component weights for mixed gaussians
numComponents = length(componentWeights); % # of components for mixed gaussians
meanVectors = Parameters.meanVectors; % Mean vectors for all gaussians
covarMatrices = Parameters.covarMatrices; % Covariance matrices for all gaussians
d = size(meanVectors,1); % Dimension determined from size of columns in mean vectors
data = zeros(d,N); % Zero-pad the data vector to d-dimension x N-samples
classLabels = (rand(1,N) >= classPriors(1)); % Create random-valued vector from 0 to 1 and check whether values are >= P(L=0)
for i = 0:numClasses-1 % For loop for class labels 0 and 1
    idx = find(classLabels==i); % logical array that finds indices of classes 0 and 1
    if i == 0 % class 0 - contains 2 Gaussians components
        n0 = length(idx); % # of Class 0 samples to generate
        mixedParameters.priors = componentWeights; % weights of gaussian components
        mixedParameters.meanVectors = meanVectors(:,1:numComponents); % component means
        mixedParameters.covarMatrices = covarMatrices(:,:,1:numComponents); % component covariance matrices
        [data(:,idx),gaussComponents] = GaussMixModel(n0,mixedParameters);  % generate mixed gaussian into data vector at indices of class 0
        plot(data(1,idx(gaussComponents==1)),data(2,idx(gaussComponents==1)),'rsquare'), hold on, 
        plot(data(1,idx(gaussComponents==2)),data(2,idx(gaussComponents==2)),'rsquare'), hold on,
    elseif i == 1 % class 1 - contains 1 single gaussian component
        n1 = length(idx); % # of Class 1 samples to generate
        m1 = meanVectors(:,3)'; % mean vector for class 1
        C1 = covarMatrices(:,:,3);
        data(:,idx) = mvnrnd(m1,C1,n1)';
        plot(data(1,idx),data(2,idx),'bo'), hold on,
        axis equal, xlabel('x_1'), ylabel('x_2'), 
        legend('p(x|L=0)', '','p(x|L=1)'),
        xticks(-5:1:9), yticks(-5:1:9)
        title('Scatter Plot of Gaussian Mixture Model'), hold off
    end
end

function [data,componentLabels] = GaussMixModel(N,Parameters)
% N = # of samples to generate
% Parameters contains mixed gaussian component parameters
priors = Parameters.priors; % Priors or weights for each component
mu = Parameters.meanVectors; % Mean vectors for each component
covarMatrix = Parameters.covarMatrices; % Covariances for each component
dim = size(mu,1); % Dimension determined from size of mean vectors
numGauss = length(priors); % # of gaussian components determined from length of weights
data = zeros(dim,N);    % Zero-pad data to d-dimensions x samples
componentLabels = zeros(1,N); % Zero-pad component labels to size of N
% Compute randomly which sample will be generated from each component
randSample = rand(1,N); t = [cumsum(priors)];
for i = 1:numGauss
    idx = find(randSample <= t(i)); % find which indices will be generated randomly for each component
    numSamples = length(idx); % # of samples to generate for component
    componentLabels(1,idx) = i*ones(1,numSamples); % Keep track of component labels
    randSample(1,idx) = 1.1*ones(1,numSamples); % these samples should not be used again
    data(:,idx) = mvnrnd(mu(:,i),covarMatrix(:,:,i),numSamples)'; % Generate component data at the random indice selected
end