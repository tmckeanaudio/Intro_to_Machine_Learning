function [data,classLabels] = HW2gmmData3D(N,Parameters)
% N = # of samples to generate
% Returns data and class labels
% Determine dimensions of data from parameters
%
rng(7);
% Gaussian Mixture Model Specifications
classPriors = Parameters.priors; % Class Priors
numClasses = length(classPriors); % # of classes
component4label = Parameters.component4label; % Vector stating # of components for each class
% Count how many components for each class there are
for i = 1:numClasses
    numComponent4eachClass(i) = sum(component4label == i);
end
componentWeights = Parameters.componentWeights; % Component weights for mixed gaussians
numComponents = length(componentWeights); % # of components for mixed gaussians
meanVectors = Parameters.meanVectors; % Mean vectors for all gaussians
covarMatrices = Parameters.covarMatrices; % Covariance matrices for all gaussians
d = size(meanVectors,1); % Dimension determined from size of columns in mean vectors
data = zeros(d,N); % Zero-pad the data vector to d-dimension x N-samples
classLabels = zeros(1,N); % Zero-pad class labels
randSample = rand(1,N); % random vector to decide # of samples 
t = [cumsum(classPriors),1]; % threshold values based on class priotrs for random sample generation
figure('Units','inches','Position',[0 0 12 8])
for i = 1:numClasses % For loop for class sample generation
    classIndex = find(randSample <= t(i));
    idx_len = length(classIndex); % # of samples randomly choosen for class
    classLabels(1,classIndex) = i*ones(1,idx_len); % store index values of class labels
    randSample(1,classIndex) = 1.1*ones(1,idx_len); % Assure choosen indices are not selected again
    if i == 1 % class 1 - contains 1 single gaussian component
        m1 = meanVectors(:,i)'; % mean vector for class 1
        C1 = covarMatrices(:,:,i); % covariance matrix for class 1
        data(:,classIndex) = mvnrnd(m1,C1,idx_len)';
        plot3(data(1,classIndex),data(2,classIndex),data(3,classIndex),'g^'), hold on,
    elseif i == 2 % class 2 - contains 1 single gaussian component
        m2 = meanVectors(:,i)'; % mean vector for class 1
        C2 = covarMatrices(:,:,i); % covariance matrix for class 1
        data(:,classIndex) = mvnrnd(m2,C2,idx_len)';
        plot3(data(1,classIndex),data(2,classIndex),data(3,classIndex),'ro'), hold on
    elseif i == 3 % class 3 - contains 2 Gaussians components
        mixedParameters.priors = componentWeights; % weights of gaussian components
        mixedParameters.meanVectors = meanVectors(:,i:i+1); % component means
        mixedParameters.covarMatrices = covarMatrices(:,:,i:i+1); % component covariance matrices
        [data(:,classIndex),gaussComponents] = GaussMixModel(idx_len,mixedParameters);  % generate mixed gaussian into data vector at indices of class 0
        plot3(data(1,classIndex(gaussComponents==1)),data(2,classIndex(gaussComponents==1)),data(3,classIndex(gaussComponents==1)),'bx'), hold on, 
        plot3(data(1,classIndex(gaussComponents==2)),data(2,classIndex(gaussComponents==2)),data(3,classIndex(gaussComponents==2)),'bx'), hold on,
        axis equal, xlabel('x_1'), ylabel('x_2'), zlabel('x_3') 
        legend('p(x|L=1)','p(x|L=2)','p(x|L=3)'),
        %xticks(-5:1:9), yticks(-5:1:9)
        title('Scatter Plot of 3D Gaussian Mixture Model'), hold off
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
