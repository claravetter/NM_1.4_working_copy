function load_selComponents(hDropdown, visData, curclass, curmod)
% load_selComponents  Populate a dropdown with kept component info (per modality).
%
% load_selComponents(hDropdown, visData, curclass, curmod)
%   hDropdown : handle to popupmenu
%   visData   : NM.analysis{x}.visdata{y}
%   curclass  : current class index
%   curmod    : current modality index

if nargin < 4, curmod = 1; end
if ~isfield(visData,'Report')
    set(hDropdown, 'String', {'<no components available>'}, ...
                           'Enable','off', 'Value',1, 'UserData',[]);
    return
end
try
    meanPval = visData.Report{curclass}.components.mean_p_cv2;
catch
    meanPval = [];
end
try
    meanCorr = visData.Report{curclass}.components.mean_r_cv2;
catch
    meanCorr = [];
end
% Pull per-modality arrays
try
    selPct = visData.Report{curclass}.components.coverage_cv1(:,1) *100 ./ visData.Report{curclass}.components.coverage_cv1(:,2);
catch
    selPct=[];
end

if isempty(meanCorr) && isempty(meanPval)
    % If there are no kept components, disable
    set(hDropdown, 'String', {'<no components available>'}, ...
                       'Enable','off', 'Value',1, 'UserData',[]);
    return
end
keptMask  = visData.CompKept{1};
idxs      = find(keptMask);
nKept     = numel(idxs);

% Means across CV2 models

% Build item list (+1 for the “stats” entry at the top)
items = cell(nKept+1, 1);
items{1} = '↗ View component statistics…';

for ii = 1:nKept

    % safe fetches in case arrays are shorter than k
    if ~isempty(selPct),   sf = selPct(ii);     end
    if ~isempty(meanCorr), cc = meanCorr(ii);   end

    if ~isempty(meanPval) && isfinite(meanPval(ii))
        pv = meanPval(ii);
        if pv < .001, ptxt = sprintf('p<%.3f', pv); else, ptxt = sprintf('p=%.3f', pv); end
        items{ii+1} = sprintf('Component #%d: pres=%1.1f%% | avg corr=%.3f | %s', idxs(ii), sf, cc, ptxt);
    else
        items{ii+1} = sprintf('Component #%d: pres=%1.1f%% | avg corr=%.3f', idxs(ii), sf, cc);
    end
end

% Map dropdown rows → component indices (row 1 = stats entry)
ud.componentIdx = [NaN, idxs(:)'];  % NaN marks the stats item
ud.modality     = curmod;
ud.class        = curclass;

set(hDropdown, 'String', items, 'Value', 2, 'Enable', 'on', 'UserData', ud);
end
