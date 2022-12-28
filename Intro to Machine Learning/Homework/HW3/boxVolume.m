% Set initial guess values for box dimensions
initLength = 1;
initWidth = 1;
initHeight = 1;

% Load initial guesses into array
x0 = [initLength initWidth initHeight];

% Call solver to minimize the objective function given the constraint
xopt = fmincon(@objective,x0,[],[],[],[],[],[],@constraint,[])

% Retrieve optimized box sizing and volume
volumeOpt = calculateVolume(xopt)

% Calculate Surface Area with optimized values just to double check
surfAreaOpt = calculateSurface(xopt)

% Define function to calculate volume of box
function vol = calculateVolume(x)
    length = x(1);
    width = x(2);
    height = x(3);
    vol = length*width*height;
end

% Define function to calculate surface area of box
function surfaceArea = calculateSurface(x)
    length = x(1);
    width = x(2);
    height = x(3);
    surfaceArea = 2*length*width + 2*length*height + 2*width*height;
end

% Define objective function for optimization
function obj = objective(x)
    obj = -calculateVolume(x);
end

% Define constraint for optimization
function [c, ceq] = constraint(x)
    c = calculateSurface(x) - 10;
    ceq = [];
end