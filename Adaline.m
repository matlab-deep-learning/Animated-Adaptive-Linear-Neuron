classdef Adaline < matlab.graphics.chartcontainer.ChartContainer
    properties(Access=public)
        alpha;
        maxSteps;
        theshold;
        batchSize;
        
        showBatch;
        showMiniBatch;
        showStochastic;
        
        pauseSec;
        outfile;                    % animation gif
        showHighlight;              % highlight dots used by minibatch/stochatic
    end
    
    properties(Access=protected)
        TrainingData;
        
        Class1Scatter;
        Class2Scatter;
        MiniBHighlight;
        StochHighlight;
        
        BatchDecision;
        StochDecision;
        MiniBDecision;
        
        CostBatchPlot;
        CostStochPlot;
        CostMiniBPlot;
        
        TopAxes;
        BotAxes;
        
        % weights
        Wbatch;
        W0batch;
        
        Wstoch;
        W0stoch;
        
        WminiB;
        W0miniB;
    end
    
    properties(Access=protected)
        Xmin;
        Xmax;
        Ymin;
        Ymax;
        
        Xlabel;
        Ylabel;
        DoneGraDes;
    end
    
    methods(Access=protected)
        function setup(obj)
            tcl = getLayout(obj);
            tcl.GridSize = [2 1];
            obj.TopAxes = nexttile(tcl, 1);
            obj.BotAxes = nexttile(tcl, 2);
            
            obj.Class1Scatter = scatter(obj.TopAxes, NaN, NaN, 'c*');
            hold(obj.TopAxes, 'on');
            obj.Class2Scatter = scatter(obj.TopAxes, NaN, NaN, 'mo');
            obj.MiniBHighlight = scatter(obj.TopAxes, NaN, NaN, 'MarkerFaceColor', 'b', 'MarkerFaceAlpha', 0.1, 'MarkerEdgeColor', 'b');
            obj.StochHighlight = scatter(obj.TopAxes, NaN, NaN, 'MarkerFaceColor', 'g', 'MarkerFaceAlpha', 0.1, 'MarkerEdgeColor', 'g');
            
            obj.BatchDecision = plot(obj.TopAxes, NaN, NaN, 'r-');
            obj.StochDecision = plot(obj.TopAxes, NaN, NaN, 'g-');
            obj.MiniBDecision = plot(obj.TopAxes, NaN, NaN, 'b-');
            hold(obj.TopAxes, 'off');
            
            obj.CostBatchPlot = plot(obj.BotAxes, NaN, NaN, 'r-');
            hold(obj.BotAxes, 'on');
            obj.CostStochPlot = plot(obj.BotAxes, NaN, NaN, 'g-');
            obj.CostMiniBPlot = plot(obj.BotAxes, NaN, NaN, 'b-');
            hold(obj.BotAxes, 'off');
        end
        
        function update(obj)
            if ~isempty(obj.Wbatch) && obj.showBatch
                obj.updateDecisionBoundary(obj.BatchDecision, obj.Wbatch, obj.W0batch);
            end
            
            if ~isempty(obj.Wstoch) && obj.showStochastic
                obj.updateDecisionBoundary(obj.StochDecision, obj.Wstoch, obj.W0stoch);
            end
            
            if ~isempty(obj.WminiB) && obj.showMiniBatch
                obj.updateDecisionBoundary(obj.MiniBDecision, obj.WminiB, obj.W0miniB);
            end
        end
        
        function initDecisionBound(obj)
            w = rand(2,1);
            w0 = rand(1);
            if obj.showBatch
                obj.Wbatch = w;
                obj.W0batch = w0;
            end
            
            if obj.showStochastic
                obj.Wstoch = w;
                obj.W0stoch = w0;
            end
            
            if obj.showMiniBatch
                obj.WminiB = w;
                obj.W0miniB = w0;
            end
        end
        
        function updateDecisionBoundary(obj, DecisionBoundary, W, W0)
            if ~isempty(obj.Ymax)
                [X, Y] = Adaline.getline(W(1), W(2), W0, [obj.Xmin obj.Xmax obj.Ymin obj.Ymax]);
                DecisionBoundary.XData = X;
                DecisionBoundary.YData = Y;
            end
        end
        
        function ret = substitute(~, X, W, W0)
            ret = X*W + W0;
        end
        
        function [Wret, W0ret] = graDescent(obj, X, y, W, W0, CostPlot)     
            ret = obj.substitute(X{:, :}, W, W0);
            errors = (y{:, :} - ret);
            Wret = W + obj.alpha*transpose(X{:, :})*errors;
            W0ret= W0 + obj.alpha*sum(errors);
            % scale it by lenght of y so as to get a fair comparison
            cost = transpose(errors)*errors/2/length(y{:, :});
            if isnan(CostPlot.XData)
                CostPlot.YData = cost;
                CostPlot.XData = 1;
            else
                CostPlot.YData = [CostPlot.YData, cost];
                CostPlot.XData = [CostPlot.XData, CostPlot.XData(end) + 1];
            end
        end
        
    end
    methods(Static)        
        % Utility functions
        function [X, Y] = getline(a, b, c, rng)
            % a line of ax+by+c=0
            XMIN=1;XMAX=2;YMIN=3;YMAX=4;
            gety = @(x) (-c-a*x)/b;

            xtmp = gety(rng(YMIN));
            if xtmp < rng(XMIN)
                X(1) = xtmp;
                Y(1) = gety(xtmp);
            else
                X(1) = rng(XMIN);
                Y(1) = gety(X(1));
            end

            xtmp = gety(rng(YMAX));
            if xtmp > rng(XMAX)
                X(2) = xtmp;
                Y(2) = gety(xtmp);
            else
                X(2) = rng(XMAX);
                Y(2) = gety(X(2));
            end
        end
    end
    
    methods(Access=public)
        function obj = Adaline(csvFile)
            
            obj.alpha = 0.005;
            obj.maxSteps = 200;
            obj.theshold = 0.00001;
            obj.batchSize = 20;
            
            obj.pauseSec = 0.2;
            
            % Preprocess data
            obj.preprocess(csvFile);
            
            % axis labels
            xlabel(obj.TopAxes, [obj.Xlabel '(scaled)']);
            ylabel(obj.TopAxes, [obj.Ylabel '(scaled)']);
            xlabel(obj.BotAxes, 'Epochs');
            ylabel(obj.BotAxes, 'Sum-squared-error');
            
            class1Data = obj.TrainingData(obj.TrainingData.cls==1, :);
            class2Data = obj.TrainingData(obj.TrainingData.cls==-1, :);
            
            obj.Class1Scatter.XData = class1Data.Var1s;
            obj.Class1Scatter.YData = class1Data.Var2s;
            
            obj.Class2Scatter.XData = class2Data.Var1s;
            obj.Class2Scatter.YData = class2Data.Var2s;
            
            obj.Xmin = min(obj.TrainingData.Var1s);
            obj.Xmax = max(obj.TrainingData.Var1s);
            obj.Ymin = min(obj.TrainingData.Var2s);
            obj.Ymax = max(obj.TrainingData.Var2s);
            xlim(obj.TopAxes, [obj.Xmin obj.Xmax]);
            ylim(obj.TopAxes, [obj.Ymin obj.Ymax]);
            
            % Default settings. Allow user to overwrite later
            obj.showBatch = true;
            obj.showStochastic = false;
            obj.showMiniBatch = false;
            obj.showHighlight = false;
            obj.outfile = [];               % no animation output by default
        end
        
        function preprocess(obj, csvFile)
            obj.TrainingData = readtable(csvFile);
            
            obj.Xlabel = obj.TrainingData.Properties.VariableNames{1};
            obj.Ylabel = obj.TrainingData.Properties.VariableNames{2};
            
            var1 = obj.TrainingData(:, obj.Xlabel).Variables;
            var1mean = mean(var1);
            var1std = std(var1);
            obj.TrainingData.Var1s = (var1 - var1mean)/var1std;
            
            var2 = obj.TrainingData(:, obj.Ylabel).Variables;
            var2mean = mean(var2);
            var2std = std(var2);
            obj.TrainingData.Var2s = (var2 - var2mean)/var2std;
        end
        function setLegend(obj)
            plots = [];
            legends = {};
            if obj.showBatch
                plots(end+1) = obj.CostBatchPlot;
                legends{end+1} = 'Batch';
            end
            if obj.showMiniBatch
                plots(end+1) = obj.CostMiniBPlot;
                legends{end+1} = 'Mini Batch';
            end
            if obj.showStochastic
                plots(end+1) = obj.CostStochPlot;
                legends{end+1} = 'Stochastic';
            end
            legend(obj.BotAxes, plots, legends);
        end
        function animate(obj)
            % init decision boundary
            obj.initDecisionBound();
            xlim(obj.BotAxes, [0 obj.maxSteps]);

            obj.setLegend();
            obj.DoneGraDes = [~obj.showBatch ~obj.showStochastic ~obj.showMiniBatch];
            
            numSamples = height(obj.TrainingData);
            firstFrame = true;
            for i=1:obj.maxSteps
                if obj.showBatch && ~obj.DoneGraDes(1)
                    [obj.Wbatch, obj.W0batch] = obj.graDescent(obj.TrainingData(:, {'Var1s', 'Var2s'}), obj.TrainingData(:, 'cls'), obj.Wbatch, obj.W0batch, obj.CostBatchPlot);

                    if length(obj.CostBatchPlot.YData) > 1
                        obj.DoneGraDes(1) = abs(obj.CostBatchPlot.YData(end) - obj.CostBatchPlot.YData(end-1)) < obj.theshold;
                    end
                end
                % minibatch gradient 
                if obj.showMiniBatch && ~obj.DoneGraDes(3)
                    % randomly pick mini batch training data
                    idx = randsample([1:numSamples], obj.batchSize);
                    batchTable = obj.TrainingData(idx, :);
                    [obj.WminiB, obj.W0miniB] = obj.graDescent(batchTable(:, {'Var1s', 'Var2s'}), batchTable(:, 'cls'), obj.WminiB, obj.W0miniB, obj.CostMiniBPlot);
                              
                    % highlight the dots being used in gradient descent
                    if obj.showHighlight
                        obj.MiniBHighlight.XData = batchTable.Var1s;
                        obj.MiniBHighlight.YData = batchTable.Var2s;
                    end
                    if length(obj.CostMiniBPlot.YData) > 1
                        obj.DoneGraDes(3) = abs(obj.CostMiniBPlot.YData(end) - obj.CostMiniBPlot.YData(end-1)) < obj.theshold;
                    end
                end
                
                if obj.showStochastic && ~obj.DoneGraDes(2)
                    % randomly pick one for stochastic batch
                    j = randi(100, 1);
                    stochTable = obj.TrainingData(j, :);
                    [obj.Wstoch, obj.W0stoch] = obj.graDescent(stochTable(:, {'Var1s', 'Var2s'}), stochTable(:, 'cls'), obj.Wstoch, obj.W0stoch, obj.CostStochPlot);
                    if obj.showHighlight
                        obj.StochHighlight.XData = stochTable.Var1s;
                        obj.StochHighlight.YData = stochTable.Var2s;
                    end
                    
                    if length(obj.CostStochPlot.YData) > 1
                        obj.DoneGraDes(2) = abs(obj.CostStochPlot.YData(end) - obj.CostStochPlot.YData(end-1)) < obj.theshold;
                    end
                end
                
                if all(obj.DoneGraDes)
                    break;
                end
                if ~isempty(obj.outfile)
                    [img, map] = rgb2ind(frame2im( getframe(gcf)),256);
                    if firstFrame
                        imwrite(img,map,obj.outfile,'gif','DelayTime',0.5);
                        firstFrame = false;
                    else
                        imwrite(img,map,obj.outfile,'gif','writemode', 'append','delaytime', obj.pauseSec);
                    end
                else
                    pause(obj.pauseSec);
                end
            end
        end
    end
end