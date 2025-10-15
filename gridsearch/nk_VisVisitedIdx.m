function vizHandles = nk_VisVisitedIdx(vizHandles, visited, candidateMatrix, hpNames)
% nk_VisVisitedIdx Optimized visualization of visited candidate indices.
%
%   vizHandles = nk_VisVisitedIdx(vizHandles, visited, candidateMatrix, hpNames)
%
%   Inputs:
%     vizHandles      - (Optional) Structure with stored figure and graphics
%                       handles. If empty or not provided, a new figure is created.
%     visited         - Structure array of visited candidate info with fields:
%                         .index : Candidate index (row in candidateMatrix)
%                         .cost  : Evaluated cost (e.g. CV1 performance)
%                         .CV2   : (Optional) CV2 performance parameter
%     candidateMatrix - Numeric matrix where each row is a candidate and each
%                       column is a hyperparameter.
%     hpNames         - (Optional) Cell array of hyperparameter names. If provided,
%                       these names will be used in the plots instead of "HP <n>".
%
%   Behavior:
%     * If the number of hyperparameters (columns with >1 unique value) is less than 5,
%       the function creates a figure with a 2-row layout for each qualifying hyperparameter.
%       The top row displays a histogram of visited values, and the bottom row shows a grouped
%       bar plot of the average cost and (if available) the average CV2 per bin.
%
%     * Otherwise (mode 2), the function creates a single pair of subplots:
%           - Left: Histogram of visited candidate indices.
%           - Right: Grouped bar plot of average cost and CV2 per candidate-index bin.
%
%   The function updates only the changing data if the figure already exists.
%
%   Example:
%       hpNames = {'Learning Rate', 'Num Trees', 'Max Depth'};
%       vizHandles = nk_VisVisitedIdx(vizHandles, DISP.visited, Ps{1}, hpNames);

    % If there are no visited entries, exit.
    if isempty(visited)
        warning('No visited candidate indices to visualize.');
        return;
    end

    % Determine if hyperparameter names were provided.
    if nargin < 4 || isempty(hpNames)
        useDefaultNames = true;
    else
        useDefaultNames = false;
    end

    % Extract visited candidate indices, costs, and CV2 (if available).
    visitedIndices = [visited.index];
    visitedCosts   = [visited.cost];
    hasCV2 = isfield(visited, 'CV2');
    if hasCV2
        visitedCV2 = [visited.CV2];
    end

    % Determine which hyperparameters vary.
    nTotal = size(candidateMatrix, 2);
    qualHP = [];
    for i = 1:nTotal
        if numel(unique(candidateMatrix(:,i))) > 1
            qualHP(end+1) = i; %#ok<AGROW>
        end
    end
    nQual = numel(qualHP);

    % Mode decision: if fewer than 5 qualifying hyperparameters, plot each individually.
    if nQual < 5
        mode = 1;
    else
        mode = 2;
    end

    % If vizHandles is not provided or the stored figure is invalid, create a new one.
    if nargin < 1 || isempty(vizHandles) || ~isfield(vizHandles, 'fig') || ~isvalid(vizHandles.fig)
        vizHandles = struct();
        figTag = 'VisitedIndicesViz';
        vizHandles.fig = figure('Tag', figTag, 'Name', 'Visited Indices Analysis','NumberTitle','off');
        
        switch mode
            case 1
                % ----- Mode 1: Individual Hyperparameter Plots -----
                vizHandles.mode = 1;
                vizHandles.hpIndices = qualHP;
                vizHandles.nPlots = nQual;
                vizHandles.axHist = cell(1, nQual);
                vizHandles.axBar  = cell(1, nQual);
                vizHandles.hHist  = cell(1, nQual);
                vizHandles.hBar   = cell(1, nQual);
                vizHandles.binEdges = cell(1, nQual);
                vizHandles.binCenters = cell(1, nQual);
                numBins = 10;
                for j = 1:nQual
                    col = qualHP(j);
                    if useDefaultNames || col > numel(hpNames)
                        paramLabel = sprintf('HP %d', col);
                    else
                        paramLabel = hpNames{col};
                    end
                    dataAll = candidateMatrix(:, col);
                    minVal = min(dataAll);
                    maxVal = max(dataAll);
                    edges = linspace(minVal, maxVal, numBins+1);
                    centers = edges(1:end-1) + diff(edges)/2;
                    vizHandles.binEdges{j} = edges;
                    vizHandles.binCenters{j} = centers;
                    hpValues = candidateMatrix(visitedIndices, col);
                    counts = histcounts(hpValues, edges);
                    axH = subplot(2, nQual, j, 'Parent', vizHandles.fig);
                    vizHandles.axHist{j} = axH;
                    hBarHist = bar(axH, centers, counts, 'FaceColor', 'flat');
                    vizHandles.hHist{j} = hBarHist;
                    xlabel(axH, sprintf('%s Value', paramLabel));
                    ylabel(axH, 'Frequency');
                    title(axH, sprintf('%s: Visited', paramLabel));
                    set(axH, 'XLim', [minVal, maxVal]);
                    
                    axB = subplot(2, nQual, nQual+j, 'Parent', vizHandles.fig);
                    vizHandles.axBar{j} = axB;
                    [~, ~, binIdx] = histcounts(hpValues, edges);
                    avgCost = nan(size(centers));
                    for b = 1:length(centers)
                        inBin = (binIdx == b);
                        if any(inBin)
                            avgCost(b) = mean(visitedCosts(inBin));
                        else
                            avgCost(b) = 0;
                        end
                    end
                    if hasCV2
                        avgCV2 = nan(size(centers));
                        for b = 1:length(centers)
                            inBin = (binIdx == b);
                            if any(inBin)
                                avgCV2(b) = mean(visitedCV2(inBin));
                            else
                                avgCV2(b) = 0;
                            end
                        end
                        dataBar = [avgCost(:), avgCV2(:)];
                        hBarAvg = bar(axB, centers, dataBar, 'grouped');
                    else
                        hBarAvg = bar(axB, centers, avgCost, 'FaceColor', 'flat');
                    end
                    vizHandles.hBar{j} = hBarAvg;
                    xlabel(axB, sprintf('%s Value', paramLabel));
                    ylabel(axB, 'Avg Performance');
                    if hasCV2
                        title(axB, sprintf('%s: Avg Cost CV1 & CV2', paramLabel));
                    else
                        title(axB, sprintf('%s: Avg Cost', paramLabel));
                    end
                    set(axB, 'XLim', [minVal, maxVal]);
                end
                
            case 2
                % ----- Mode 2: Overall Candidate Index Plots -----
                vizHandles.mode = 2;
                nCandidates = size(candidateMatrix, 1);
                uniqueVisited = unique(visitedIndices);
                if numel(uniqueVisited) < 10
                    numBins = numel(uniqueVisited);
                else
                    numBins = 10;
                end
                edges = linspace(0.5, nCandidates+0.5, numBins+1);
                centers = edges(1:end-1) + diff(edges)/2;
                vizHandles.binEdges = edges;
                vizHandles.binCenters = centers;
                
                counts = histcounts(visitedIndices, edges);
                [~, ~, binIdx] = histcounts(visitedIndices, edges);
                avgCost = nan(size(centers));
                for b = 1:length(centers)
                    inBin = (binIdx == b);
                    if any(inBin)
                        avgCost(b) = mean(visitedCosts(inBin));
                    else
                        avgCost(b) = 0;
                    end
                end
                if hasCV2
                    avgCV2 = nan(size(centers));
                    for b = 1:length(centers)
                        inBin = (binIdx == b);
                        if any(inBin)
                            avgCV2(b) = mean(visitedCV2(inBin));
                        else
                            avgCV2(b) = 0;
                        end
                    end
                end
                
                vizHandles.axLeft = subplot(1,2,1, 'Parent', vizHandles.fig);
                vizHandles.hLeft = bar(vizHandles.axLeft, centers, counts, 'FaceColor', 'flat');
                xlabel(vizHandles.axLeft, 'Candidate Index');
                ylabel(vizHandles.axLeft, 'Frequency');
                title(vizHandles.axLeft, 'Visited Candidate Indices');
                set(vizHandles.axLeft, 'XLim', [min(centers), max(centers)]);
                
                vizHandles.axRight = subplot(1,2,2, 'Parent', vizHandles.fig);
                if hasCV2
                    dataBar = [avgCost(:), avgCV2(:)];
                    vizHandles.hRight = bar(vizHandles.axRight, centers, dataBar, 'grouped');
                    title(vizHandles.axRight, 'Avg Cost CV1 & CV2 per Candidate Bin');
                else
                    vizHandles.hRight = bar(vizHandles.axRight, centers, avgCost, 'FaceColor', 'flat');
                    title(vizHandles.axRight, 'Avg Cost per Candidate Bin');
                end
                xlabel(vizHandles.axRight, 'Candidate Index Bin');
                ylabel(vizHandles.axRight, 'Avg Performance');
                set(vizHandles.axRight, 'XLim', [min(centers), max(centers)]);
        end
        sgtitle(vizHandles.fig, 'Visited Indices Analysis');
        drawnow;
        
    else
        % The figure exists: update only the bar objects.
        switch vizHandles.mode
            case 1
                numBins = 10;
                for j = 1:vizHandles.nPlots
                    col = vizHandles.hpIndices(j);
                    if useDefaultNames || col > numel(hpNames)
                        paramLabel = sprintf('HP %d', col);
                    else
                        paramLabel = hpNames{col};
                    end
                    edges = vizHandles.binEdges{j};
                    centers = vizHandles.binCenters{j};
                    minVal = edges(1);
                    maxVal = edges(end);
                    
                    hpValues = candidateMatrix(visitedIndices, col);
                    counts = histcounts(hpValues, edges);
                    [~, ~, binIdx] = histcounts(hpValues, edges);
                    avgCost = nan(size(centers));
                    for b = 1:length(centers)
                        inBin = (binIdx == b);
                        if any(inBin)
                            avgCost(b) = mean(visitedCosts(inBin));
                        else
                            avgCost(b) = 0;
                        end
                    end
                    if hasCV2
                        avgCV2 = nan(size(centers));
                        for b = 1:length(centers)
                            inBin = (binIdx == b);
                            if any(inBin)
                                avgCV2(b) = mean(visitedCV2(inBin));
                            else
                                avgCV2(b) = 0;
                            end
                        end
                        dataBar = [avgCost(:), avgCV2(:)];
                        axes(vizHandles.axBar{j});
                        cla(vizHandles.axBar{j});
                        vizHandles.hBar{j} = bar(vizHandles.axBar{j}, centers, dataBar, 'grouped');
                        xlabel(vizHandles.axBar{j}, sprintf('%s Value', paramLabel));
                        ylabel(vizHandles.axBar{j}, 'Avg Performance');
                        title(vizHandles.axBar{j}, sprintf('%s: Avg CV1 & CV2', paramLabel));
                    else
                        set(vizHandles.hBar{j}, 'YData', avgCost, 'XData', centers);
                    end
                    set(vizHandles.hHist{j}, 'YData', counts, 'XData', centers);
                    set(vizHandles.axHist{j}, 'XLim', [minVal, maxVal]);
                    set(vizHandles.axBar{j}, 'XLim', [minVal, maxVal]);
                end
            case 2
                nCandidates = size(candidateMatrix,1);
                uniqueVisited = unique(visitedIndices);
                if numel(uniqueVisited) > 10 
                    numBins = numel(uniqueVisited);
                else
                    numBins = 10;
                end
                edges = linspace(0.5, nCandidates+0.5, numBins+1);
                centers = edges(1:end-1) + diff(edges)/2;
                counts = histcounts(visitedIndices, edges);
                [~, ~, binIdx] = histcounts(visitedIndices, edges);
                avgCost = nan(size(centers));
                for b = 1:length(centers)
                    inBin = (binIdx == b);
                    if any(inBin)
                        avgCost(b) = mean(visitedCosts(inBin));
                    else
                        avgCost(b) = 0;
                    end
                end
                if hasCV2
                    avgCV2 = nan(size(centers));
                    for b = 1:length(centers)
                        inBin = (binIdx == b);
                        if any(inBin)
                            avgCV2(b) = mean(visitedCV2(inBin));
                        else
                            avgCV2(b) = 0;
                        end
                    end
                    dataBar = [avgCost(:), avgCV2(:)];
                    axes(vizHandles.axRight);
                    cla(vizHandles.axRight);
                    vizHandles.hRight = bar(vizHandles.axRight, centers, dataBar, 'grouped');
                    title(vizHandles.axRight, 'Avg CV1 & CV2 per Candidate Bin');
                else
                    set(vizHandles.hRight, 'YData', avgCost, 'XData', centers);
                end
                % Update the left bar plot using the stored bar handle (hLeft).
                set(vizHandles.hLeft, 'YData', counts, 'XData', centers);
                set(vizHandles.axLeft, 'XLim', [min(centers), max(centers)]);
                set(vizHandles.axRight, 'XLim', [min(centers), max(centers)]);
        end
        sgtitle(vizHandles.fig, 'Visited Indices Analysis');
        drawnow;
    end

end
