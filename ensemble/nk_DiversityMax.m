function [opt_hE, opt_E, opt_F, opt_D] = nk_DiversityMax(E, L, EnsStrat)
% =========================================================================
% [opt_hE, opt_E, opt_F, opt_D] = nk_DiversityMax(E, L, EnsStrat)
% =========================================================================
% Greedy ensemble constructor that optimizes a *pure diversity* objective
% (no performance in the criterion). Useful for decorrelating errors.
%
% Search modes via EnsStrat.type:
%   • type==1 : Backward Elimination (remove worst wrt diversity)
%   • type==5 or 7 : Forward Selection (seed by best pair, then grow)
%
% INPUTS / OUTPUTS
% ----------------
% Same E, L as nk_CVMax. EnsStrat adds:
%   .DiversityObjective 'A' (double-fault), 'Q' (Yule’s Q), or 'K' (Fleiss κ)
%                       (default 'Q')
%   .MaxNum             max #learners to keep (default k)
%   .DiversityEps       ε tie-tolerance on diversity (default 0.02)
%   .SizeReward         γ ≥ 0 size reward used as: J(S)=D(S)−γ·log|S|
%   .MinNum             recommended ≥3 for pairwise statistics
%
% Returns:
%   opt_hE : performance of the selected set (reported only)
%   opt_E  : selected columns of E
%   opt_F  : selected indices
%   opt_D  : minimized value of the diversity objective:
%            'A' → A   (lower better), 'Q' → Q (lower better),
%            'K' → κ   (lower better)
%
% NOTES
% -----
% • For 'Q'/'A' the function uses nk_Diversity; for 'K' uses nk_DiversityKappa.
% • Forward mode seeds with the best pair by the chosen objective.
% • If the optimized set is not better than the original (by diversity),
%   the full set is returned.
%
% SEE ALSO
% --------
% nk_MultiDiversityMax, nk_Diversity, nk_DiversityKappa, nk_Entropy, nk_Lobag
% =========================================================================
% (c) Nikolaos Koutsouleris, 10/2025

global VERBOSE

% ---------- sanitize ----------
ind0 = (L ~= 0); E = E(ind0,:); L = L(ind0);
[~,k] = size(E);
if k == 0
    opt_hE = 0; opt_E = E; opt_F = []; opt_D = NaN; return;
end

% ---------- defaults ----------
if ~isfield(EnsStrat,'MinNum')      || isempty(EnsStrat.MinNum),      EnsStrat.MinNum = 2; end
if ~isfield(EnsStrat,'MaxNum')      || isempty(EnsStrat.MaxNum),      EnsStrat.MaxNum = k; end
if ~isfield(EnsStrat,'DiversityEps')|| isempty(EnsStrat.DiversityEps),EnsStrat.DiversityEps = 0.02; end
if ~isfield(EnsStrat,'SizeReward')  || isempty(EnsStrat.SizeReward),  EnsStrat.SizeReward = 0; end

if isfield(EnsStrat,'DiversitySource') && ~isempty(EnsStrat.DiversitySource)
    switch lower(EnsStrat.DiversitySource)
        case 'entropy', obj = 'H';
        case 'kappaa',  obj = 'A';
        case 'kappaq',  obj = 'Q';
        case 'kappaf',  obj = 'K';
        case 'lobag',   obj = 'ED';
        otherwise,      obj = 'Q';
    end
else
    obj = 'Q';
end
if ismember(obj, {'A','Q','K'}), EnsStrat.MinNum = max(EnsStrat.MinNum, 2); end

% ---------- hard predictions used for diversity ----------
if isfield(EnsStrat,'Metric') && EnsStrat.Metric==2
    T = sign(E); T(T==0) = -1;    % NM convention for dichotomizers
else
    T = E;                        % already hard labels {-1,+1}
end

% ---------- scoring helper: f(subset idx) -> scalar (to MINIMIZE) ----------
    function s = score_idx(idx)
        if isempty(idx), s = Inf; return; end
        P = T(:, idx);                      % <- slice the ensemble here

        switch obj
            case 'H'   % maximize vote-entropy ⇒ minimize negative entropy
                Ph = sign(P); Ph(Ph==0) = -1;
                s = -nk_Entropy(Ph, [-1 1], size(Ph,2), []);

            case 'A'     % double-fault (lower is better); needs ≥2 learners
                if size(P,2) < 2, s = Inf; return; end
                [A,~] = nk_Diversity(P, L, [], []);
                s = A;

            case 'Q'     % Yule’s Q (more negative is better); needs ≥2 learners
                if size(P,2) < 2, s = Inf; return; end
                [~,Q] = nk_Diversity(P, L, [], []);
                s = isfinite(Q) * Q + ~isfinite(Q) * 0; % neutral fallback 0

            case 'K'     % Fleiss κ (lower is better); needs ≥2 learners
                if size(P,2) < 2, s = Inf; return; end
                D1mK = nk_DiversityKappa(P, L, [], []); % returns 1-κ
                s = 1 - D1mK;                           % = κ

            case 'ED'    % LoBag ED (lower is better) – use hard labels
                Ph = sign(P); Ph(Ph==0) = -1;
                s = nk_Lobag(Ph, L);

            otherwise
                if size(P,2) < 2, s = Inf; return; end
                [~,Q] = nk_Diversity(P, L, [], []);
                s = isfinite(Q) * Q;
        end
        
    end

% ---------- baselines (for report only) ----------
opt_hE = nk_EnsPerf(E, L);  orig_hE = opt_hE;
opt_D  = score_idx(1:k);    orig_D  = opt_D;
orig_F = 1:k;

% ---------- search ----------
switch EnsStrat.ConstructMode
    case 1   % Backward elimination (min diversity objective)
        strhdr = sprintf('BE => min(%s)', obj);
        I = orig_F; opt_I = I; iter=0; kcur = k; Best = opt_D;

        while kcur > EnsStrat.MinNum
            lD = inf(kcur,1);
            for l = 1:kcur
                kI = I; kI(l) = [];
                lD(l) = score_idx(kI, L);
            end
            [cand, l] = min(lD);

            % remove the index that gives the smallest objective
            I(l) = []; kcur = kcur - 1;

            if cand <= Best + eps
                Best = cand; opt_I = I; iter = iter + 1;
            end
        end
        MinParam = Best;

    case 2  % Forward selection (min diversity objective), perf-free
        label = obj; if obj=='K', label='Kappa'; end
        strhdr = sprintf('FS => min(%s)', label);

        if k < 2
            opt_I = 1; MinParam = score_idx(1); iter = 0;
        else
            origI = 1:k;

            % seed with best pair
            bestD = Inf; best_pair = [];
            for a = 1:k-1
                for b = a+1:k
                    val = score_idx([a b]);
                    if isfinite(val) && val < bestD
                        bestD = val; best_pair = [a b];
                    end
                end
            end
            if isempty(best_pair)
                opt_I = 1:k; MinParam = score_idx(1:k); iter = 0;
                I = opt_I;    % for final consistency
            else
                I        = best_pair;
                MinParam = bestD;
                opt_I    = I;
                iter     = 1;
            end

            % remove seeded pair from pool
            origI(ismember(origI, I)) = [];
            kleft = numel(origI);

            % penalized objective J(S) = D(S) - γ*log(|S|)
            gamma  = EnsStrat.SizeReward;
            Jbest  = MinParam - gamma*log(numel(I));

            % grow while candidates remain and we haven't hit MaxNum
            while kleft > 0 && numel(I) < EnsStrat.MaxNum
                lD = inf(kleft,1);
                for l = 1:kleft
                    kI   = [I, origI(l)];
                    valD = score_idx(kI);
                    if isfinite(valD), lD(l) = valD; end
                end

                [candD, l] = min(lD);
                if ~isfinite(candD)
                    origI(l) = []; kleft = kleft - 1;
                    continue
                end

                % Accept if penalized objective improves OR
                % raw objective is within ε of best (and we still want to grow),
                % or we haven’t met MinNum yet and within relaxed ε.
                m_cur  = numel(I);
                Jcand  = candD - gamma*log(m_cur + 1);
                accept = (Jcand <= Jbest - 1e-12) || ...
                         (candD <= MinParam + EnsStrat.DiversityEps) || ...
                         (m_cur < EnsStrat.MinNum && candD <= MinParam + 10*EnsStrat.DiversityEps);

                if accept
                    I        = [I, origI(l)];
                    MinParam = min(MinParam, candD);
                    Jbest    = min(Jbest, Jcand);
                    opt_I    = I; iter = iter + 1;
                end

                origI(l) = []; kleft = kleft - 1;
            end

            % if still below MinNum, force-fill with next best
            while numel(I) < EnsStrat.MinNum && ~isempty(origI)
                lD = inf(numel(origI),1);
                for l = 1:numel(origI)
                    lD(l) = score_idx([I, origI(l)]);
                end
                [candD, l] = min(lD);
                if ~isfinite(candD), origI(l) = []; continue; end
                I = [I, origI(l)]; opt_I = I; iter = iter + 1;
                origI(l) = [];
            end
        end

    otherwise
        error('EnsStrat.type must be 1 (BE) or 5/7 (FS).');
end

% ---------- finalize ----------
% keep the better (smaller) objective
if opt_D <= MinParam + eps
    % keep full
    opt_D = orig_F; opt_F = orig_F; opt_E = E;
    opt_hE = nk_EnsPerf(opt_E, L);
else
    opt_F = opt_I; opt_E = E(:, opt_I);
    opt_hE = nk_EnsPerf(opt_E, L);
    opt_D = MinParam;
end

if VERBOSE
   fprintf(['\n%s: %g iters\t' ...
    'Div(%s, orig->final): %1.3f->%1.3f, ' ...
    'Perf (orig->final): %1.3f->%1.3f, ' ...
    '# Learners (orig->final): %g->%g'], ...
    strhdr, iter, obj, orig_D, opt_D, nk_EnsPerf(E,L), opt_hE, numel(orig_F), numel(opt_F));
end
end