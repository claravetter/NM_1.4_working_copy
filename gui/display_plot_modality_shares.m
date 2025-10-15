function display_plot_modality_shares(handles, varargin)
%DISPLAY_PLOT_MODALITY_SHARES  Interactive plot for L2n share reports.
% Views (from ReportFinal):
%   1) Modality-wise      -> Report{h}.modality
%   2) Component-wise     -> Report{h}.components
%   3) Component×Modality -> Report{h}.comp_by_mod
%
% Usage:
%   display_plot_modality_shares(handles, h)
%   display_plot_modality_shares(handles, h, 'TopK', 20)   % for component-wise view
%
% Requirements:
%   - handles.I2.Report{h} exists (see 'report_final' block in nk_VisXHelperC)
%
% UI:
%   - Drop-down (top-left) to choose view
%   - Export button (top-right) to save current view as .xlsx (Windows) or .csv (elsewhere)

%% Parse options
p = inputParser;
p.addParameter('TopK', 15, @(x)isnumeric(x) && isscalar(x) && x>=1);
p.parse(varargin{:});
TopK = p.Results.TopK;

%% Fetch report
if ~isfield(handles, 'visdata') || ~isfield(handles.visdata{1}, 'Report') 
    errordlg('No report data available. Generate it first.','No report data');
    return;
end
R = handles.visdata{1}.Report{handles.curclass};

%% Build figure and UI controls
f = figure('Name','L2n shares — views', 'NumberTitle','off', 'Color','w', ...
           'Units','normalized','Position',[0.15 0.15 0.7 0.7]);

% Axes area (centered a bit more, more top margin)
ax = axes('Parent', f, 'Units','normalized', 'Position', [0.08 0.14 0.62 0.74]); %#ok<NASGU>

% Table area
uitableHandle = uitable('Parent', f, 'Units','normalized', ...
    'Position',[0.72 0.14 0.26 0.74], ...
    'Data',{}, 'ColumnName',{}, 'RowName',{}, 'Visible','off');

% View selector (more space above axes; left-top)
if ~isfield(R,'components')
    viewNames = {'Modality-wise (median ± 95% CI)'};
    compwise = false; compflag = 'off';
else
    viewNames = { ...
       'Modality-wise (median ± 95% CI)', ...
       'Component-wise L2 shares (Median Top-K)', ...
       'Component-wise L2 shares (Mean Top-K)', ...
       'Component-wise mean r (CV2) ± 95% CI (Top-K)', ...
       'Component-wise mean p (CV2) ± 95% CI (Top-K)', ...
       'Component × Modality (median share)'};
    compwise = true; compflag = 'on';
end

popupView = uicontrol('Style','popupmenu', 'String',viewNames, ...
    'Units','normalized', 'Position',[0.02 0.93 0.56 0.055], ...
    'FontSize',10, 'BackgroundColor',[0.95 0.95 0.95], 'Enable', compflag);

% Export button (stays to the right; avoid overlap)
btnExport = uicontrol('Style','pushbutton','String','Export current view…', ...
    'Units','normalized','Position',[0.82 0.93 0.16 0.055], ...
    'FontSize',10,'BackgroundColor',[0.9 0.95 1]);

if compwise
    btnSettings = uicontrol('Style','pushbutton','String','⚙ Settings…', ...
        'Units','normalized','Position',[0.64 0.93 0.16 0.055], ...
        'FontSize',10,'BackgroundColor',[0.95 0.95 0.95]);
    btnSettings.Callback = @(src,evt) showSettingsDialog(f);
end

% Store state in figure appdata
S = struct();
S.compwise = compwise;
S.R       = R;
S.TopK    = TopK;
S.handles = handles;
setappdata(f, 'state', S);

% Hook callbacks
popupView.Callback = @(src,evt) redrawCurrent(f, popupView, uitableHandle); 
btnExport.Callback = @(src,evt) exportCurrent(f, popupView);                

% Initial draw
redrawCurrent(f, popupView, uitableHandle);

movegui(f,'onscreen'); drawnow; figure(f);  % bring to front once

end % main function

%% =======================================================================
function redrawCurrent(figHandle, popupView, uitableHandle)
cla; ax = gca;
S = getappdata(figHandle, 'state');
R = S.R;
set(ax, 'YLimMode','auto', 'CLimMode','auto');  % reset on every redraw
viewIdx = popupView.Value;

switch viewIdx
    case 1
        % ===================== Modality-wise ============================
        M = R.modality;
        if isempty(M) || ~isfield(M,'median') || isempty(M.median)
            text(0.5,0.5,'No modality-wise data','HorizontalAlignment','center'); axis off; 
            set(uitableHandle,'Visible','off');
            return;
        end

        % Prefer summary with 95% CI if present (lo95/hi95 optional)
        med  = M.median(:);
        names= M.names(:);
        nM   = numel(med);
        
        % Use precomputed CIs if present; otherwise none
        haveCI = isfield(M,'lo95') && isfield(M,'hi95') && numel(M.lo95)==nM && numel(M.hi95)==nM;
        if haveCI
            lo95 = M.lo95(:); hi95 = M.hi95(:);
        end
        
        bar(ax, med, 'FaceAlpha', 0.9); hold(ax,'on');
        if haveCI
            x = 1:nM; y = med;
            neg = max(0, y - lo95);
            pos = max(0, hi95 - y);
            errorbar(ax, x, y, neg, pos, 'k', 'LineStyle','none', 'LineWidth', 1.0);
        end
        
        % Export table includes CI if present
        T = table(names, med, 'VariableNames',{'Modality','MedianShare'});
        if isfield(M,'mean') && numel(M.mean)==nM, T.MeanShare = M.mean(:); end
        if haveCI, T.CI_lo95 = lo95; T.CI_hi95 = hi95; end
        if isfield(M,'coverage') && size(M.coverage,1)==nM
            T.CoverageHits = M.coverage(:,1); T.CoverageTotal= M.coverage(:,2);
        end
        if isfield(S,'colorbar'), S.colorbar.Visible='off'; end
        S.current.table   = T;
        S.current.matrix  = table2cell(T);
        S.current.headers = T.Properties.VariableNames;
        S.current.name    = sprintf('ModalityShares');
        setappdata(figHandle,'state',S);
        if strcmp(get(uitableHandle,'Visible'),'off')
            set(ax,'Position',[0.08 0.14 0.86 0.74]); % wider when no table
        else
            set(ax,'Position',[0.08 0.14 0.62 0.74]); % default
        end
        ax.XAxis.TickLabels = names;
        ax.YAxis.Label.String = 'Modality importance (L_p share)';
        % compute upper target from median/CI, clamp to [0,1]
        if exist('lo95','var') && exist('hi95','var') && ~isempty(lo95) && ~isempty(hi95)
            set_lim_share(ax, y, lo95, hi95);
        else
            set_lim_share(ax, y);
        end

    case {2,3}
        % ===================== Component-wise ===========================
        % Ensure the table is visible and shrink the axes so they don't overlap (even on first entry)
        set(uitableHandle,'Visible','on');
        ax = gca; set(ax,'Position',[0.08 0.14 0.62 0.74]);
    
        C = R.components;
        if isempty(C) || ~isfield(C,'median_share') || isempty(C.median_share)
            text(0.5,0.5,'No component-wise data','HorizontalAlignment','center'); axis off;
            set(uitableHandle,'Visible','off'); return;
        end
        med   = C.median_share(:);
        meanV = C.mean_share(:);
        names = C.names(:);
        nRef  = numel(med);
        ord   = C.order_by_median(:);
        if isempty(ord), [~,ord] = sort(med,'descend','MissingPlacement','last'); end

        K = min(getappdata(figHandle,'state').TopK, nRef);
        pick = ord(1:K);

        % Bar plot of median shares (top-K)
        switch viewIdx
            case 2
                vals = med(pick);
                tit = sprintf('Component-wise L2 shares (Top-%d by median)', K);
                ylab = 'L2 share (median)';
            case 3
                vals = meanV(pick);
                tit = sprintf('Component-wise L2 shares (Top-%d by mean)', K);
                ylab = 'L2 share (mean)';
        end
        bar(ax, vals, 'FaceAlpha', 0.9);
        set(ax, 'XTick', 1:K, 'XTickLabel', names(pick), 'XTickLabelRotation', 30);
        ylabel(ax, ylab);
        title(ax, tit);
        ax.YGrid='on'; ax.Box='on'; ax.FontSize=11;
        set_lim_share(ax, vals(pick));

        % Build a side table with stats (mean p/r and presence)
        T = table(names(pick), med(pick), meanV(pick), 'VariableNames',{'Component','MedianShare','MeanShare'});
        if isfield(C,'mean_p_cv2'),  T.MeanP_CV2  = C.mean_p_cv2(pick);  end
        if isfield(C,'mean_r_cv2'),  T.MeanR_CV2  = C.mean_r_cv2(pick);  end
        if isfield(C,'present_cv2') && isfield(C,'denom_cv2')
            pres = C.present_cv2(pick); den = C.denom_cv2(pick);
            T.Presence = arrayfun(@(a,b) sprintf('%d/%d', a, max(1,b)), pres, den, 'uni',0);
        end
        % coverage across CV1 folds if present
        if isfield(C,'coverage_cv1') && size(C.coverage_cv1,1) >= nRef
            T.CV1Covg = arrayfun(@(a,b) sprintf('%d/%d', a, b), ...
                          C.coverage_cv1(pick,1), C.coverage_cv1(pick,2), 'uni',0);
        end
        if isfield(S,'colorbar'), S.colorbar.Visible='off'; end

        % Show table
        set(uitableHandle, ...
            'Data',        table2cell(T), ...
            'ColumnName',  T.Properties.VariableNames, ...
            'RowName',     [], ...
            'Visible',     'on', ...
            'ColumnEditable', false(1, width(T)));

        % Stash for export
        S.current.table   = T;
        S.current.matrix  = table2cell(T);   
        S.current.headers = T.Properties.VariableNames;
        S.current.name    = sprintf('ComponentShares_h%03d_Top%d', R.meta.h, K);
        setappdata(figHandle,'state',S);

    case 4 % Correlation coefficients
        C = R.components;
        if isempty(C) || ~isfield(C,'mean_r_cv2') || isempty(C.mean_r_cv2)
            text(0.5,0.5,'No component-wise CV2 correlation data','HorizontalAlignment','center'); axis off;
            set(uitableHandle,'Visible','off'); return;
        end
        
        names = C.names(:);
        rmean = C.mean_r_cv2(:);
        rlo   = []; rhi = [];
        if isfield(C,'mean_r_lo95'), rlo = C.mean_r_lo95(:); end
        if isfield(C,'mean_r_hi95'), rhi = C.mean_r_hi95(:); end
    
        nRef  = numel(rmean);
        % Rank by median L2 share (keeps same ordering anchor as your main L2 report)
        ord = C.order_by_median; if isempty(ord), [~,ord] = sort(C.median_share,'descend','MissingPlacement','last'); end
        K = min(getappdata(figHandle,'state').TopK, nRef); pick = ord(1:K);
    
        bar(ax, rmean(pick), 'FaceAlpha', 0.9); hold(ax,'on');
        if ~isempty(rlo) && ~isempty(rhi) && numel(rlo)==nRef && numel(rhi)==nRef
            x = 1:K; y = rmean(pick);
            neg = max(0, y - rlo(pick)); pos = max(0, rhi(pick) - y);
            errorbar(ax, x, y, neg, pos, 'k', 'LineStyle','none', 'LineWidth', 1.0);
        end
        set(ax, 'XTick', 1:K, 'XTickLabel', names(pick), 'XTickLabelRotation', 30);
        ylabel(ax, 'mean r (CV2)'); title(ax, sprintf('Component-wise mean correlation (Top-%d)', K));
        ax.YGrid='on'; ax.Box='on'; ax.FontSize=11;
        if exist('rlo','var') && exist('rhi','var') && ~isempty(rlo) && ~isempty(rhi)
            set_lim_corr(ax, rmean(pick), rlo(pick), rhi(pick));
        else
            set_lim_corr(ax, rmean(pick));
        end
    
        % Side table
        T = table(names(pick), rmean(pick), 'VariableNames',{'Component','MeanR_CV2'});
        if ~isempty(rlo) && ~isempty(rhi)
            T.MeanR_lo95 = rlo(pick); T.MeanR_hi95 = rhi(pick);
        end
        if isfield(C,'present_cv2') && isfield(C,'denom_cv2')
            T.Presence = arrayfun(@(a,b) sprintf('%d/%d', a, max(1,b)), C.present_cv2(pick), C.denom_cv2(pick), 'uni',0);
        end
        set(uitableHandle,'Data', table2cell(T), 'ColumnName', T.Properties.VariableNames, 'RowName',[], 'Visible','on');
        if isfield(S,'colorbar'), S.colorbar.Visible='off'; end
        if strcmp(get(uitableHandle,'Visible'),'off')
            set(ax,'Position',[0.08 0.14 0.86 0.74]); % wider when no table
        else
            set(ax,'Position',[0.08 0.14 0.62 0.74]); % default
        end

        S.current.table   = T;
        S.current.matrix  = table2cell(T);
        S.current.headers = T.Properties.VariableNames;
        S.current.name    = sprintf('Component_MeanR_CV2_Top%d', K);
        setappdata(figHandle,'state',S);

    case 5 % P values
        C = R.components;
        if isempty(C) || ~isfield(C,'mean_p_cv2') || isempty(C.mean_p_cv2)
            text(0.5,0.5,'No component-wise CV2 p-value data','HorizontalAlignment','center'); axis off;
            set(uitableHandle,'Visible','off'); return;
        end
        names = C.names(:);
        pmean = C.mean_p_cv2(:);
        plo   = []; phi = [];
        if isfield(C,'mean_p_lo95'), plo = C.mean_p_lo95(:); end
        if isfield(C,'mean_p_hi95'), phi = C.mean_p_hi95(:); end
    
        nRef  = numel(pmean);
        ord = C.order_by_median; if isempty(ord), [~,ord] = sort(C.median_share,'descend','MissingPlacement','last'); end
        K = min(getappdata(figHandle,'state').TopK, nRef); pick = ord(1:K);
    
        bar(ax, pmean(pick), 'FaceAlpha', 0.9); hold(ax,'on');
        if ~isempty(plo) && ~isempty(phi) && numel(plo)==nRef && numel(phi)==nRef
            x = 1:K; y = pmean(pick);
            neg = max(0, y - plo(pick)); pos = max(0, phi(pick) - y);
            errorbar(ax, x, y, neg, pos, 'k', 'LineStyle','none', 'LineWidth', 1.0);
        end
        set(ax, 'XTick', 1:K, 'XTickLabel', names(pick), 'XTickLabelRotation', 30);
        ylabel(ax, 'mean -log10(p) (CV2)'); title(ax, sprintf('Component-wise mean p (Top-%d)', K));
        ax.YGrid='on'; ax.Box='on'; ax.FontSize=11;
        if exist('plo','var') && exist('phi','var') && ~isempty(plo) && ~isempty(phi)
            set_lim_logp(ax, pmean(pick), plo(pick), phi(pick));
        else
            set_lim_logp(ax, pmean(pick));
        end
    
        T = table(names(pick), pmean(pick), 'VariableNames',{'Component','MeanP_CV2'});
        if ~isempty(plo) && ~isempty(phi)
            T.MeanP_lo95 = plo(pick); T.MeanP_hi95 = phi(pick);
        end
        if isfield(C,'present_cv2') && isfield(C,'denom_cv2')
            T.Presence = arrayfun(@(a,b) sprintf('%d/%d', a, max(1,b)), C.present_cv2(pick), C.denom_cv2(pick), 'uni',0);
        end
        set(uitableHandle,'Data', table2cell(T), 'ColumnName', T.Properties.VariableNames, 'RowName',[], 'Visible','on');
        if isfield(S,'colorbar'), S.colorbar.Visible='off'; end
        if strcmp(get(uitableHandle,'Visible'),'off')
            set(ax,'Position',[0.08 0.14 0.86 0.74]); % wider when no table
        else
            set(ax,'Position',[0.08 0.14 0.62 0.74]); % default
        end
        S.current.table   = T;
        S.current.matrix  = table2cell(T);
        S.current.headers = T.Properties.VariableNames;
        S.current.name    = sprintf('Component_MeanP_CV2_Top%d', K);
        setappdata(figHandle,'state',S);

    case 6 % Heat map
        % ===================== Component × Modality =====================
        CM = R.comp_by_mod;
        if isempty(CM) || ~isfield(CM,'median_share') || isempty(CM.median_share)
            text(0.5,0.5,'No component×modality data','HorizontalAlignment','center'); axis off;
            set(uitableHandle,'Visible','off');
            return;
        end
    
        M = CM.median_share;         % <-- define the matrix in-scope
        compNames = CM.names_comp(:);
        modNames  = CM.names_mod(:);
    
        hImg = imagesc(M, 'Parent', ax); axis(ax,'tight');
        colormap(ax, parula);
        if isfield(S,'colorbar')
            S.colorbar.Visible='off'; 
        else
            S.colorbar=colorbar('peer', ax);
        end
        set(ax, 'XTick', 1:numel(modNames), 'XTickLabel', modNames, 'XTickLabelRotation', 30);
        set(ax, 'YTick', 1:numel(compNames), 'YTickLabel', compNames);
        xlabel(ax,'Modality'); ylabel(ax,'Component');
        title(ax, sprintf('Component × Modality median share'));
        set(ax,'Position',[0.18 0.14 0.60 0.74]); % wider when no table

        % Adaptive color scale in [0,1]
        cmax = max(M(:), [], 'omitnan');
        if isfinite(cmax) && cmax > 0
            clim(ax, [0 min(1, cmax*1.02)]);
        else
            clim(ax, [0 1]);
        end
    
        set(uitableHandle,'Visible','off');
    
        % Stash for export (numeric matrix is fine)
        T = array2table(M, 'VariableNames', matlab.lang.makeValidName(modNames), ...
                           'RowNames', compNames);
        S.current.table   = T;
        S.current.matrix  = M;
        S.current.headers = [{'Component'}, modNames(:)'];
        S.current.name    = sprintf('CompByMod_median_h%03d', R.meta.h);
        setappdata(figHandle,'state',S);
end

end

%% =======================================================================
function exportCurrent(figHandle, popupView) 
% Export the current view to XLSX (Windows) or CSV (elsewhere)
S = getappdata(figHandle, 'state');
if ~isfield(S,'current') || isempty(S.current)
    warndlg('Nothing to export yet.','Export');
    return;
end

% sensible default: .xlsx on Windows, else .csv
defaultExt  = ternary(ispc, '.xlsx', '.csv');
defaultName = [S.current.name, defaultExt];

[fn, fp, fi] = uiputfile( ...
    {'*.xlsx','Excel Workbook (*.xlsx)'; '*.csv','CSV (comma delimited) (*.csv)'}, ...
    'Export current view', defaultName);

if isequal(fn,0), return; end
fullpath = fullfile(fp,fn);
[~,~,ext] = fileparts(fullpath);
ext = lower(ext);

try
    T = ensureTable(S.current.table);
    settingsTable = [];
    if isfield(S.R,'meta') && isfield(S.R.meta,'settings') && ~isempty(S.R.meta.settings)
        settingsTable = settings_as_table(S.R.meta.settings);
    end
    switch ext
        case '.xlsx'
            writetable(T, fullpath, 'WriteRowNames', hasRowNames(T), 'Sheet','Data');
            if ~isempty(settingsTable)
                writetable(settingsTable, fullpath, 'WriteRowNames', false, 'Sheet','Settings');
            end
    
        case '.csv'
            writetable(T, fullpath, 'WriteRowNames', hasRowNames(T), 'FileType','text','Delimiter',',');
            if ~isempty(settingsTable)
                [fp0,fn0] = fileparts(fullpath);
                fullpath2 = fullfile(fp0, [fn0 '_settings.csv']);
                writetable(settingsTable, fullpath2, 'WriteRowNames', false, 'FileType','text','Delimiter',',');
            end

        otherwise
            % Fallback: best-effort CSV if user typed a different extension
            fullpath = [fullpath, '.csv'];
            writetable(T, fullpath, ...
                'WriteRowNames', hasRowNames(T), ...
                'FileType', 'text', ...
                'Delimiter', ',');
    end

    msgbox(sprintf('Exported to:\n%s', fullpath), 'Export successful','help');

catch ME
    errordlg(sprintf('Export failed:\n%s', ME.message), 'Export error');
end
end

function y = ternary(cond, a, b)
% simple local ternary
if cond, y = a; else, y = b; end
end

%% ---------- helpers for export ----------
function tf = hasRowNames(T)
try
    tf = ~isempty(T.Properties.RowNames);
catch
    tf = false;
end
end

function T = ensureTable(Tin)
% Convert cell/data to table if needed; keep row names if present
if istable(Tin)
    T = Tin;
else
    % if it's raw matrix with headers stored separately, coerce to table
    if iscell(Tin)
        T = cell2table(Tin);
    else
        T = array2table(Tin);
    end
end
end

function set_lim_share(ax, y, lo, hi)
% Shares are in [0,1]. Pad upwards by 5% of the max (respecting CI if given).
    if nargin < 3 || isempty(lo) || isempty(hi)
        ymax = max(y, [], 'omitnan');
    else
        ymax = max(hi, [], 'omitnan');  % use CI upper if available
    end
    if ~isfinite(ymax) || ymax <= 0, ymax = 0.02; end
    ylim(ax, [0, min(1, ymax*1.05)]);
    xlim(ax, [0.5, numel(y)+0.5]);
    ax.YAxis.TickValuesMode = 'auto';
    ax.YAxis.TickLabelsMode= 'auto';
end

function set_lim_corr(ax, y, lo, hi)
% Correlations are in [-1,1]. Pad by 5% of span but clamp to [-1,1].
    yall = y(:);
    if nargin >= 3 && ~isempty(lo) && ~isempty(hi)
        yall = [yall; lo(:); hi(:)];
    end
    yall = yall(isfinite(yall));
    if isempty(yall), ylim(ax, [-1 1]); return; end
    ymin = min(yall); ymax = max(yall);
    pad  = 0.05 * max(1e-6, ymax - ymin);
    ylim(ax, [max(-1, ymin - pad), min(1, ymax + pad)]);
    xlim(ax, [0.5, numel(y)+0.5]);
    ax.YAxis.TickValuesMode = 'auto';
    ax.YAxis.TickLabelsMode= 'auto';
end

function set_lim_logp(ax, y, lo, hi)
% y (and optional lo/hi) are on the -log10 scale, i.e., unbounded above and >= 0.
    yall = y(:);
    if nargin >= 3 && ~isempty(lo) && ~isempty(hi)
        yall = [yall; lo(:); hi(:)];
    end
    yall = yall(isfinite(yall));
    if isempty(yall), ylim(ax,[0 1]); return; end
    ymax = max(yall);
    if ~isfinite(ymax) || ymax <= 0, ymax = 0.02; end
    pad  = 0.05 * max(1e-6, ymax);
    ylim(ax, [0, ymax + pad]);
    xlim(ax, [0.5, numel(y)+0.5]);
    ax.YAxis.TickValuesMode = 'auto';
    ax.YAxis.TickLabelsMode= 'auto';
end

function showSettingsDialog(figHandle)
S = getappdata(figHandle,'state');
if ~isfield(S,'R') || ~isfield(S.R,'meta') || ~isfield(S.R.meta,'settings') || isempty(S.R.meta.settings)
    warndlg('No settings found in R.meta.settings.','Settings');
    return;
end

T = settings_as_table(S.R.meta.settings);

% Build a small modal dialog
d = dialog('Name','Backprojection & Alignment Settings','Units','normalized', ...
           'Position',[0.32 0.32 0.36 0.36], 'Color','w');
[Data, ColNames] = table_to_uitable_data(T);

uit = uitable('Parent',d,'Units','normalized','Position',[0.05 0.18 0.90 0.77], ...
    'Data', Data, 'ColumnName', ColNames, ...
    'RowName',[], 'ColumnEditable', false(1,width(T)));

nCols = numel(ColNames);
set(uit, 'ColumnWidth', repmat({'auto'}, 1, nCols));  

uicontrol('Parent',d,'Style','pushbutton','String','Copy as text', ...
    'Units','normalized','Position',[0.05 0.05 0.28 0.09], ...
    'Callback', @(~,~) copySettingsToClipboard(T));

uicontrol('Parent',d,'Style','pushbutton','String','Close', ...
    'Units','normalized','Position',[0.67 0.05 0.28 0.09], ...
    'Callback', @(~,~) close(d));
end

function copySettingsToClipboard(T)
% Turn the table into nicely formatted "Name: Value" lines.
lines = strcat(T.Parameter, ": ", T.Value);
str = strjoin(lines, newline);
try
    clipboard('copy', str);
    msgbox('Settings copied to clipboard.','Copied','help');
catch ME
    errordlg(['Copy failed: ' ME.message],'Copy error');
end
end

function T = settings_as_table(Sin)
% Map internal field names to friendly labels + descriptions
nameMap = struct( ...
  'backprojection_method',       {{'Backprojection method', 'How component weights are projected back to feature space'}}, ...
  'similarity_alignment_method', {{'Similarity metric (alignment)', 'Metric used when aligning components (e.g., correlation type)'}}, ...
  'similarity_pruning_cutoff',   {{'Similarity cutoff', 'Minimum similarity to keep a component during alignment'}}, ...
  'presence_pruning_cutoff',     {{'Presence cutoff', 'Minimum presence (e.g., across folds) to retain a component'}} );

params = fieldnames(Sin);
P = strings(numel(params),1);
V = strings(numel(params),1);
D = strings(numel(params),1);

for i = 1:numel(params)
    fn  = params{i};
    val = Sin.(fn);

    if isfield(nameMap, fn)
        c  = nameMap.(fn);        % <-- pull the cell array out first
        nm = c{1};
        ds = c{2};
    else
        nm = fn;
        ds = "";
    end

    P(i) = string(nm);
    V(i) = stringifyValue(val);
    D(i) = string(ds);
end

T = table(P, V, D, 'VariableNames', {'Parameter','Value','Description'});
end


function s = stringifyValue(v)
% Short, readable values for numbers/strings/logicals/arrays
if isstring(v) || ischar(v)
    s = string(v);
elseif islogical(v) && isscalar(v)
    s = string(v);
elseif isnumeric(v) && isscalar(v)
    s = string(num2str(v, '%g'));
elseif isnumeric(v) && ~isscalar(v)
    sz = size(v);
    head = num2str(v(1:min(numel(v),5)),' %g'); head = strtrim(head);
    s = sprintf('[%s] %s', strjoin(string(sz),'×'), head);
elseif iscellstr(v) || (iscell(v) && all(cellfun(@ischar,v)))
    s = "{" + strjoin(string(v), ", ") + "}";
else
    try
        s = string(jsonencode(v));
    catch
        s = "<unprintable>";
    end
end
end

function [C, colnames] = table_to_uitable_data(T)
% Convert a table to a cell array where each element is numeric/logical/char.

    C = table2cell(T);
    for r = 1:size(C,1)
        for c = 1:size(C,2)
            v = C{r,c};

            if isstring(v)                 % string -> char
                C{r,c} = char(v);
            elseif ischar(v)               % ok
                % no-op
            elseif isnumeric(v) || islogical(v)
                if isscalar(v)
                    % ok
                else
                    % non-scalar numeric/logical -> summarize as char
                    C{r,c} = char(strjoin(string(v(:).'), ' '));
                end
            elseif isdatetime(v)
                C{r,c} = datestr(v);
            elseif iscell(v)               % cell -> char summary
                try
                    C{r,c} = char(strjoin(string(v), ', '));
                catch
                    C{r,c} = '<cell>';
                end
            else                            % anything else -> char via json or fallback
                try
                    C{r,c} = char(jsonencode(v));
                catch
                    C{r,c} = '<unprintable>';
                end
            end
        end
    end

    colnames = cellstr(T.Properties.VariableNames);
end
