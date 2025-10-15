function D = nk_PrintCV2BarChart(D, multiflag)
global OCTAVE

WinTag = 'PrintCVBarsBin';

%% --- Figure Creation / Update (Static) ---
if isfield(D, 'h') && ishandle(D.h)
    h = D.h;
    set(0, 'CurrentFigure', h);
else
    h = findobj('Tag', WinTag);
    if isempty(h)
        h = figure('Name', D.binwintitle, ...
            'NumberTitle', 'off', ...
            'Tag', WinTag, ...
            'MenuBar', 'none', ...
            'Position', D.figuresz, ...
            'Color', [0.9 0.9 0.9]);
    end
    D.h = h;
    set(0, 'CurrentFigure', h);
end
set(h, 'Name', D.binwintitle);

%% --- Progress Bar Axes (Static Creation, Dynamic Update) ---
if ~isfield(D, 'hl') || ~ishandle(D.hl)
    hl = findobj(h, 'Tag', 'CurrParam');
    if isempty(hl)
        D.hl = axes('Parent', h, 'Position', [0.1 0.95 0.85 0.025], ...
            'Tag', 'CurrParam', 'Visible', 'on', 'YTick', []);
    else
        D.hl = hl(1);  % If more than one exists, pick the first.
    end
end
set(h, 'CurrentAxes', D.hl);

% --- Progress Bar Data Update ---  
% Instead of clearing the axes each time, update only the dynamic bar object.
if ~isfield(D, 'hl_bar') || ~ishandle(D.hl_bar)
    % Create the progress bar (a single horizontal bar)
    D.hl_bar = barh(D.hl, 1, D.pltperc, 'FaceColor', 'b');
else
    % Update the bar's value; for a horizontal bar, the bar length is given by its XData.
    set(D.hl_bar, 'XData', D.pltperc);
end
hold(D.hl, 'on');

% --- Build the Progress Bar Label (Dynamic) ---
if ~isempty(D.Pdesc{1})
    tx = sprintf('%s\nParams: ', D.s);
    txi = '';
    for j = 1:numel(D.Pdesc{1})
        if iscell(D.P{1}(j))
            ParVal = D.P{1}{j};
        else
            ParVal = D.P{1}(j);
        end
        if isnumeric(ParVal)
            ParVal = num2str(ParVal, '%g');
        end
        txi = [txi sprintf('%s = %s, ', D.Pdesc{1}{j}, ParVal)];
    end
    tx = [tx txi(1:end-2)]; % Remove trailing comma.
else
    tx = D.s;
end

% Update or create the xlabel for the progress bar axes.
if ~isfield(D, 'hl_xlabel') || ~ishandle(D.hl_xlabel)
    D.hl_xlabel = xlabel(D.hl, tx);
else
    set(D.hl_xlabel, 'String', tx);
end

%% --- Axes for Bar Plots (Static Creation, Dynamic Update) ---
for p = 1:numel(D.ax)
    if ~isfield(D.ax{p}, 'h') || ~ishandle(D.ax{p}.h)
        % Static creation: create axes only once.
        ha = findobj(h, 'Tag', D.ax{p}.title);
        if isempty(ha)
            D.ax{p}.h = axes('Parent', h, 'Position', D.ax{p}.position, 'Tag', D.ax{p}.title);
        else
            D.ax{p}.h = ha(1);
        end
        % Set static properties (layout, labels, ticks, title, etc.)
        set(D.ax{p}.h, 'FontWeight', 'bold', 'FontSize', 10);
        box(D.ax{p}.h, 'on');
        ylabel(D.ax{p}.h, D.ax{p}.ylb);
        if isempty(D.ax{p}.ylm)
            ylm = [min(D.ax{p}.val_y(:)), max(D.ax{p}.val_y(:))];
        else
            ylm = D.ax{p}.ylm;
        end
        ylim(D.ax{p}.h, ylm);
        if D.nclass > 1 && size(D.ax{p}.val_y,1) == 2
            xmax = 1;
        else
            xmax = size(D.ax{p}.val_y,1);
        end
        xlim(D.ax{p}.h, [0.5, xmax+0.5]);
        set(D.ax{p}.h, 'XTick', 1:numel(D.ax{p}.label), 'XTickLabel', D.ax{p}.label);
        title(D.ax{p}.h, D.ax{p}.title);
        
        % Create the bar and error plots and store their handles.
        [D.ax{p}.h_bar, D.ax{p}.err_bar] = barwitherr(D.ax{p}.std_y, D.ax{p}.val_y);
        if D.nclass == 1
            set(D.ax{p}.h_bar, 'FaceColor', D.ax{p}.fc);
        end
        if ~isempty(D.ax{p}.lg)
            h_legend = legend(D.ax{p}.h, D.ax{p}.lg);
            legend(D.ax{p}.h, 'boxoff');
            set(h_legend, 'FontSize', 8);
        end
    else
        % Dynamic update: update only the data values.
        set(h, 'CurrentAxes', D.ax{p}.h);
        for i = 1:D.nclass
            if any(isnan(D.ax{p}.val_y(:, i)))
                continue;
            end
            if D.nclass == 1 
                std_y = D.ax{p}.std_y;
                val_y = D.ax{p}.val_y;
            else
                std_y = D.ax{p}.std_y(:, i);
                val_y = D.ax{p}.val_y(:, i);
            end
            set(D.ax{p}.h_bar(i), 'YData', val_y);
            set(D.ax{p}.err_bar(i), 'YData', val_y);
            if ~OCTAVE
                set(D.ax{p}.err_bar(i), 'YNegativeDelta', std_y);
                set(D.ax{p}.err_bar(i), 'YPositiveDelta', std_y);
            end
            if ~isempty(std_y)
                set(D.ax{p}.err_bar(i), 'LData', std_y);
                set(D.ax{p}.err_bar(i), 'UData', std_y);
            end
        end
    end
end

%% --- Additional Multi-Axes (Static Creation, Dynamic Update) ---
if multiflag
    for p = 1:numel(D.m_ax)
        if ~isfield(D.m_ax{p}, 'h') || ~ishandle(D.m_ax{p}.h)
            hm = findobj(h, 'Tag', D.m_ax{p}.title);
            if isempty(hm)
                D.m_ax{p}.h = axes('Parent', h, 'Position', D.m_ax{p}.position, ...
                    'Tag', D.m_ax{p}.title, 'FontWeight', 'bold');
                box(D.m_ax{p}.h, 'on');
            else
                D.m_ax{p}.h = hm(1);
            end
            ylabel(D.m_ax{p}.h, D.m_ax{p}.ylb);
            if isempty(D.m_ax{p}.ylm)
                ylm = [min(D.m_ax{p}.val_y(:)), max(D.m_ax{p}.val_y(:))];
            else
                ylm = D.m_ax{p}.ylm;
            end
            ylim(D.m_ax{p}.h, ylm);
            xlim(D.m_ax{p}.h, [0.5, length(D.m_ax{p}.val_y)+0.5]);
            set(D.m_ax{p}.h, 'XTick', 1:numel(D.m_ax{p}.label), 'XTickLabel', D.m_ax{p}.label);
            title(D.m_ax{p}.h, D.m_ax{p}.title);
            
            [D.m_ax{p}.h_bar, D.m_ax{p}.err_bar] = barwitherr(D.m_ax{p}.std_y, D.m_ax{p}.val_y);
            if D.nclass == 1
                set(D.m_ax{p}.h_bar, 'FaceColor', D.m_ax{p}.fc);
            end
        else
            set(h, 'CurrentAxes', D.m_ax{p}.h);
            set(D.m_ax{p}.h_bar, 'YData', D.m_ax{p}.val_y);
            set(D.m_ax{p}.err_bar, 'YData', D.m_ax{p}.val_y);
            set(D.m_ax{p}.err_bar, 'LData', D.m_ax{p}.std_y);
            set(D.m_ax{p}.err_bar, 'UData', D.m_ax{p}.std_y);
        end
    end
end

drawnow;
