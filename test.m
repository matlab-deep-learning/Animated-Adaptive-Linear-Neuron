% Example 1. Get started
ada = Adaline('sample_data.csv');
ada.animate();

% Example 2. Mini batch
ada = Adaline('sample_data.csv');
ada.showBatch = false;          % skip batch gradient descent
ada.showMiniBatch = true;       % show minibatch
ada.showStochastic = false;     % skip stochastic 
ada.maxSteps = 100;             % maximum iteration of gradient descent
ada.showHighlight = true;       % highlight mini batch samples
ada.animate();

% Example 3. Use Big alpha learning rate
ada = Adaline('sample_data.csv');
ada.alpha = 0.0155;             % a big learning rate
ada.maxSteps = 30;              % minibatch needs more iterations
ada.animate();

% Example 4. Mini batch size
ada = Adaline('sample_data.csv');
ada.showBatch = false;          % skip batch gradient descent
ada.showMiniBatch = true;       % show minibatch
ada.showStochastic = false;     % skip stochastic 
ada.batchSize = 10;             % mini batch size 
ada.showHighlight = true;       % highlight mini batch samples
ada.animate();

% Example 5 all three gradient descent
ada = Adaline('sample_data.csv');
ada.alpha = 0.005;              % set learning rate
ada.maxSteps = 100;             % minibatch needs more iterations
ada.showBatch = true;           % show batch gradient descent
ada.showMiniBatch = true;       % show minibatch
ada.showStochastic = true;      % show stochastic 
ada.showHighlight = true;       % highlight the subgroup used for gra des
ada.outfile = 'allGraDes.gif';  % output an animation gif
ada.animate();
