function display_modality_component_stats(visData, curclass, curmod, showRaw)
% display_modality_component_stats  Visualize modality-wise component stats,
% including component-specific L2 shares.
%
% Usage:
%   display_modality_component_stats(visData, curclass, curmod)
%   display_modality_component_stats(visData, curclass, curmod, showRaw)
%
% Inputs:
%   visData   : struct holding per-modality visualization data
%   curclass  : class index (1-based)
%   curmod    : modality index (1-based)
%   showRaw   : (optional, default=false) overlay raw presence (%) in panel 1
%
% Expected fields (backward compatible cell shapes):
%   CorrRef_CV2{curclass,curmod} or {curmod}
%   PermPValRef_CV2{...} (optional)
%   CompKept{...} (optional, logical)
%   ModComp_L2n{...} (optional)              % [nComp x nCV1] or [nComp x 1]
%   ModComp_L2n_SUMMARY{...}.median/lo95/hi95/n (optional)

if nargin < 4, showRaw = false; end

% --- fetch core inputs ----------------------------------------------------
corrMat = pick_cell(visData, 'CorrRef_CV2',      curclass, curmod);
pvalMat = pick_cell(visData, 'PermPValRef_CV2',  curclass, curmod);
kept    = pick_cell(visData, 'CompKept',         curclass, curmod);
if ~isempty(kept), kept = kept(:); end

if isempty(corrMat)
    warning('display_modality_component_stats:empty', ...
        'No CorrRef_CV2 available for class %d, modality %d.', curclass, curmod);
    return
end

% --- presence metrics -----------------------------------------------------
[~, excessPct, zScore] = display_presence_ratio(corrMat);
nComp = numel(excessPct);
x     = 1:nComp;

% --- mean correlation & (optional) permutation p-values -------------------
meanCorr = mean(corrMat, 2, 'omitnan');
hasPerm  = ~isempty(pvalMat);
if hasPerm
    mlogP = -log10(mean(pvalMat, 2, 'omitnan'));
end

% --- component-specific L2 shares ----------------------------------------
haveShares = false; useSummary = false;
medShare = []; loShare = []; hiShare = []; nShare = [];
Sshare = pick_cell(visData, 'ModComp_L2n', curclass, curmod);

% Prefer precomputed summaries if available
SUM = pick_cell(visData, 'ModComp_L2n_SUMMARY', curclass, curmod);
if isstruct(SUM) && all(isfield(SUM, {'median','lo95','hi95','n'}))
    medShare = SUM.median(:);
    loShare  = SUM.lo95(:);
    hiShare  = SUM.hi95(:);
    nShare   = SUM.n(:);
    if ~isempty(medShare)
        % clamp to available components
        r = min(numel(medShare), nComp);
        medShare = medShare(1:r); loShare = loShare(1:r); hiShare = hiShare(1:r); nShare = nShare(1:r);
        haveShares = true; useSummary = true;
    end
elseif ~isempty(Sshare)
    % compute summaries on the fly from raw shares
    if isvector(Sshare)
        Sshare = Sshare(:); % [nComp x 1]
    end
    if size(Sshare,1) ~= nComp
        % align to available components
        r = min(size(Sshare,1), nComp);
        Sshare = Sshare(1:r, :);
        nComp  = r; x = 1:nComp; excessPct = excessPct(1:r); presPct = presPct(1:r);
        meanCorr = meanCorr(1:r);
        if hasPerm, mlogP = mlogP(1:r); end
        if ~isempty(kept), kept = kept(1:r); end
    end
    medShare = nm_nanmedian(Sshare, 2);
    loShare  = nm_nanquantile(Sshare', 0.025)';   % empirical 95% CI
    hiShare  = nm_nanquantile(Sshare', 0.975)';
    nShare   = sum(isfinite(Sshare), 2);
    haveShares = true;
end

% --- figure layout (add a tile when shares exist) -------------------------
nRows = 2 + hasPerm + haveShares;
figTitle = sprintf('Modality %d – Component Stats', curmod);
fig = figure('Name', figTitle, 'Color','w', 'NumberTitle','off');
tl  = tiledlayout(nRows, 1, 'Padding','compact', 'TileSpacing','compact');

% 1) Adjusted presence (%), optional raw overlay
nexttile;
bar(x, excessPct, 'FaceAlpha', 0.9); hold on;
yline(100, ':', '100% expected', 'HandleVisibility','off');
ylabel('Propensity-adjusted excess presence [%]');
xlabel('Ref component #');
title('Presence across CV2 (adjusted)');
ylim([-100 100]);
xlim([0.5, nComp + 0.5]);

if showRaw
    plot(x, presPct, '-', 'LineWidth', 1.0, 'DisplayName','raw %');
    legend({'adjusted %','raw %'}, 'Location','northeast');
end

% highlight kept comps
if ~isempty(kept) && numel(kept) == nComp
    kk = find(kept);
    plot(kk, excessPct(kk), 'k.', 'MarkerSize', 12);
end
hold off;

% 2) Mean correlation to reference
nexttile;
bar(x, meanCorr, 'FaceAlpha', 0.9);
ylabel('Mean corr(ref)');
xlabel('Ref component #');
title('Mean Correlation to Reference');
xlim([0.5, nComp + 0.5]);

% 3) Permutation p-values (if present)
if hasPerm
    nexttile;
    bar(x, mlogP, 'FaceAlpha', 0.9);
    ylabel('-log_{10} p (mean)');
    xlabel('Ref component #');
    title('Permutation p-values (mean across CV2)');
    yline(-log10(0.05), '--', '0.05', 'HandleVisibility','off');
    xlim([0.5, nComp + 0.5]);
end

% 4) Component-specific L2 shares (if present)
if haveShares
    nexttile;
    hold on;
    % Bars at median
    bar(x, medShare, 'FaceAlpha', 0.85);
    % Error bars for 95% CI
    neg = max(0, medShare - loShare);
    pos = max(0, hiShare  - medShare);
    errorbar(x, medShare, neg, pos, 'k', 'LineStyle','none', 'LineWidth', 1.0);

    % If we have raw share matrix, jitter individual points
    if exist('Sshare','var') && ~isempty(Sshare) && size(Sshare,2) > 1
        rng(17);
        for i = 1:nComp
            yi = Sshare(i, :);
            yi = yi(isfinite(yi));
            if isempty(yi), continue; end
            jitter = 0.04*randn(size(yi));
            scatter(i + jitter, yi, 16, 'filled', 'MarkerFaceAlpha', 0.6, 'MarkerEdgeAlpha', 0);
        end
    end

    ylabel('L2 share (fraction)');
    xlabel('Ref component #');
    title(sprintf('Component-specific L2 shares (%s)', tern(useSummary,'precomputed','empirical')));
    ylim([0, min(1, max(0.05, max(hiShare)*1.05))]);
    xlim([0.5, nComp + 0.5]);

    % annotate n per component
    for i = 1:nComp
        text(i, max(medShare(i)+pos(i), 0.02), sprintf('n=%d', nShare(i)), ...
            'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',8);
    end
    hold off;
end

% Bring figure forward
figure(fig);

end

% ============================ HELPERS ====================================

function cellVal = pick_cell(visData, fieldName, curclass, curmod)
% Try common shapes for stored cells: {class,mod} → {mod} → {class} → {1}
cellVal = [];
if ~isfield(visData, fieldName) || isempty(visData.(fieldName)), return; end
C = visData.(fieldName);
if ~iscell(C) || isempty(C), return; end
[r,c] = size(C);
if r >= curclass && c >= curmod && ~isempty(C{curclass,curmod})
    cellVal = C{curclass,curmod};
elseif c >= curmod && ~isempty(C{curmod})
    cellVal = C{curmod};
elseif r >= curclass && ~isempty(C{curclass})
    cellVal = C{curclass};
elseif ~isempty(C{1})
    cellVal = C{1};
end
end
