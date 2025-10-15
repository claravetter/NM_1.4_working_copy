function param = nk_RegAmbig(sE, tE, mode, agg)
% =========================================================================
% nk_RegAmbig  —  Regression ensemble diversity/ambiguity
% -------------------------------------------------------------------------
% param = nk_RegAmbig(sE, tE, mode, agg)
%
% sE   : N x n matrix of base learners' predictions
% tE   : (optional) N x 1 target vector, ONLY used when mode='corr'
% mode : 'var'      -> mean per-sample variance across learners (default)
%        'pairwise' -> mean per-sample average pairwise squared difference
%        'corr'     -> 1 - |corr(ensemble_pred, tE)|  (NOT a diversity of the
%                      ensemble per se; include only if you explicitly want it)
% agg  : 'mean' (default) or 'median' ensemble aggregate (for 'var'/'pairwise')
%
% Returns:
%   param (scalar): higher = more dispersion among base predictions.
%
% Notes:
%   • 'var' matches the Krogh–Vedelsby ambiguity (with 'mean' aggregate).
%   • 'pairwise' equals (2n/(n-1)) times the 'var' (population) per sample,
%     but avoids choosing an aggregate explicitly.
%   • We do NOT include tE in the dispersion unless mode='corr'.
% =========================================================================
% (c) Nikolaos Koutsouleris, 10/2025

if nargin < 3 || isempty(mode), mode = 'var'; end
if nargin < 4 || isempty(agg),  agg  = 'mean'; end
if nargin < 2, tE = []; end

[N,n] = size(sE);
if n <= 1 || N == 0
    param = 0; 
    return;
end

switch lower(mode)
    case 'var'
        % Choose ensemble aggregate
        switch lower(agg)
            case 'median'
                mu = median(sE, 2);  % robust, breaks exact VK decomposition
            otherwise % 'mean'
                mu = mean(sE, 2);
        end
        R = sE - mu;                          % N x n residuals
        v = mean(R.^2, 2);                    % per-sample population variance across learners
        param = mean(v);                      % average over samples

    case 'pairwise'
        % Average pairwise squared difference per sample:
        % (2/(n*(n-1))) * sum_{i<j} (f_i - f_j)^2  ==  (2n/(n-1)) * popVar
        mu = mean(sE, 2);
        R = sE - mu;
        popVar = mean(R.^2, 2);               % population (not sample) variance across learners
        param = mean( (2*n/(n-1)) * popVar ); % scalar

    case 'corr'
        % Not an intra-ensemble diversity: this is a perf-like complement.
        if isempty(tE) || numel(tE) ~= N
            error('nk_RegAmbig: mode=''corr'' requires tE as N-by-1 target vector.');
        end
        ens_pred = mean(sE, 2);               % or median(sE,2) if you prefer
        c = corr(ens_pred, tE, 'type', 'Pearson', 'rows','pairwise');
        if ~isfinite(c), c = 0; end
        param = 1 - abs(c);

    otherwise
        error('nk_RegAmbig: unknown mode ''%s''.', mode);
end
end
