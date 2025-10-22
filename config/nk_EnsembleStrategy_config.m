function param = nk_EnsembleStrategy_config(param, SVM, MODEFL, defaultsfl, parentstr)

% ----------------------- defaults -----------------------------------------
Ensemble.type            = 0;
Ensemble.Metric          = 2;
Ensemble.Mode            = 3;              % hard-wired for Probabilistic Feature Extraction
Ensemble.MinNum          = 1;
Ensemble.Perc            = 75;
Ensemble.DataType        = 2;
Ensemble.Weighting       = 0;
Ensemble.DivCrit         = 2;               % 1 entropy-only, 2 perf+diversity, 3 kappa-only, 4 lobag-only, 5 reg-only
Ensemble.ConstructMode   = 0;               % 0 agg, 1 BE, 2 FS, 4 prob-subspace, 5 AdaBoost (TBD)
Ensemble.DivStr          = '';
Ensemble.DivFunc         = '';
Ensemble.OptFunc         = '';
Ensemble.EntropRegMode   = 1;               % 1 strict, 2 mixed
Ensemble.CompFunc        = 'max';
Ensemble.DiversitySource = 'entropy';       % 'entropy'|'kappaa'|'kappaq'|'kappaf'|'lobag'|'regvar'

% Regularizers (used in nk_CVMax; visible only for DivCrit==2)
Ensemble.PerfSlackPct    = 0.0;             % fraction of metric range (e.g., 0.005 = 0.5%)
Ensemble.EntropyWeight   = 2.0;
Ensemble.SizePenalty     = 0.0;
Ensemble.Patience        = 0;
SubSpaceStrategy = 1;
SubSpaceCrit     = 0;
act              = 0;
CostFun          = 2;

if nargin < 4 || isempty(defaultsfl), defaultsfl = 0; end

% ----------------------- derive max/min from SVM --------------------------
if ~isempty(SVM) && isfield(SVM,'GridParam')
    switch SVM.GridParam
        case {1,5,6,7,10,13,14,15,17}
            Ensemble.CompFunc = 'max';
        otherwise
            Ensemble.CompFunc = 'min';
    end
else
    warndlg('Setup of prediction algorithm and main optimization parameter is required!');
    return
end

% ----------------------- carry over existing settings ---------------------
if ~defaultsfl
    if isfield(param,'SubSpaceStrategy'), SubSpaceStrategy = param.SubSpaceStrategy; end
    if isfield(param,'SubSpaceCrit'),     SubSpaceCrit     = param.SubSpaceCrit;     end
    if isfield(param,'CostFun'),          CostFun          = param.CostFun;          end
    if isfield(param,'EnsembleStrategy') && isfield(param.EnsembleStrategy,'DataType')
        Ensemble = local_overwrite_defaults(Ensemble, param.EnsembleStrategy, MODEFL);
    else
        param.EnsembleStrategy = Ensemble;
    end
    if ~CostFun, param.CostFun = 2; end

    d = nk_GetParamDescription2([],param,'EnsType'); menuact = [];

    menustr = sprintf('Select subspace selection strategy [ %s ]', d.SubSpaceStrat); menuact = [menuact 1];

    if SubSpaceStrategy > 1
        menustr = sprintf('%s|Define ensemble optimization method [ %s ]', menustr, d.EnsConMode); menuact = [menuact 2];

        if Ensemble.ConstructMode && Ensemble.ConstructMode ~= 4
            % ----------------- OVERHAULED MENU (act=3) -------------------
            if strcmpi(MODEFL,'classification')
                menustr = sprintf('%s|Select the optimization function [ %s ]', menustr, local_divcrit_label(Ensemble));
            else
                menustr = sprintf('%s|Select the optimization function [ %s ]', menustr, local_divcrit_label_reg(Ensemble));
            end
            menuact = [menuact 3];

            % If combining perf+diversity, expose individual regularizers:
            if Ensemble.DivCrit == 2
                menustr = sprintf('%s|Choose diversity metric to combine with performance [ %s ]', ...
                                  menustr, local_pretty_divsrc(Ensemble.DiversitySource));        menuact = [menuact 9];
                menustr = sprintf('%s|Set performance slack (%% of criterion range) [ %g %% ]', ...
                                  menustr, 100*Ensemble.PerfSlackPct);                            menuact = [menuact 10];
                menustr = sprintf('%s|Set diversity weight (rank blend w_E) [ %g ]', ...
                                  menustr, Ensemble.EntropyWeight);                               menuact = [menuact 11];
                menustr = sprintf('%s|Set size penalty λ (0=off) [ %g ]', ...
                                  menustr, Ensemble.SizePenalty);                                 menuact = [menuact 12];
                menustr = sprintf('%s|Set early-stop patience (steps, 0=off) [ %g ]', ...
                                  menustr, Ensemble.Patience);                                    menuact = [menuact 13];
                menustr = sprintf('%s|Show quick help on regularizers', menustr);                 menuact = [menuact 14];
            else
                menustr = sprintf('%s|Set diversity epsilon ε (tolerance) [ %g ]', ...
                                  menustr, Ensemble.DiversityEps);                                menuact = [menuact 15];
                menustr = sprintf('%s|Set size penalty γ (0=off) [ %g ]', ...
                                  menustr, Ensemble.SizePenalty);                                 menuact = [menuact 12];
                menustr = sprintf('%s|Show quick help on diversity-only knobs', menustr);         menuact = [menuact 16];
            end
        end

        if Ensemble.type ~= 0 && Ensemble.type ~= 9
            if strcmpi(MODEFL,'classification')
                menustr = sprintf('%s|Use algorithm output scores or label predictions [ %s ]', menustr, d.FeatMetric); menuact = [menuact 4];
            else
                Ensemble.Metric = 1; Ensemble.DiversitySource = 'regvar';
            end
            menustr = sprintf('%s|Define minimum # of classifiers to be selected [ %1.0f ]', menustr, Ensemble.MinNum);  menuact = [menuact 5];
            menustr = sprintf('%s|Enable weighting of feature subspaces [ %s ]', menustr, d.EnsWeighting);              menuact = [menuact 6];
        elseif Ensemble.type == 9
            menustr = sprintf('%s|Define percentage of cross-subspace feature agreement [ %g ]', menustr, Ensemble.Perc);     menuact = [menuact 7];
            menustr = sprintf('%s|Define minimum number of features to select across feature subspaces [ %1.0f ]', menustr, Ensemble.MinNum); menuact = [menuact 8];
            Ensemble.Weighting = 0;
        end
    end

    nk_PrintLogo
    mestr = 'Subspace-based ensemble optimization setup'; fprintf('\nYou are here: %s >>>',parentstr);
    act = nk_input(mestr, 0,'mq', menustr, menuact);

    switch act
        case 1
            switch SVM.GridParam
                case {9,11,12,18}, mxstr='minimum '; threshstr='BELOW'; df=25;
                otherwise,          mxstr='maximum'; threshstr='ABOVE'; df=75;
            end
            SubSpaceStrategy = nk_input('Subspace selection strategy',0, 'm', ...
                ['Subspace with ' mxstr d.CostType ' criterion (winner takes it all)|' ...
                 'Subspace ensemble with ' mxstr d.CostType ' within a range of the maximum|' ...
                 'Subspace ensemble ' mxstr d.CostType ' ' threshstr ' a given percentile|' ...
                 'All-subspace ensemble'],1:4, SubSpaceStrategy);
            switch SubSpaceStrategy
                case 1, SubSpaceCrit = 0; Ensemble.ConstructMode = 0;
                case 2, SubSpaceCrit = nk_input(['Range from ' d.CostType mxstr],0,'e', 5);
                case 3, SubSpaceCrit = nk_input(['Percentile [%] for ' d.CostType ' cutoff'],0,'e', df);
                case 4, SubSpaceCrit = 0; Ensemble.ConstructMode = 0;
            end

        case 2
            Ensemble.ConstructMode = nk_input('Select ensemble construction method',0,'m', ...
                ['Simply aggregate all learners into ensemble|' ...
                 'Optimize ensemble using backward base learner elimination|' ...
                 'Optimize ensemble using forward base learner selection|' ...
                 'Create a single classifier using probabilistic feature subspace construction|' ...
                 'Use AdaBoost to determine optimal weighting of bases learners'],[0 1 2 4 5], Ensemble.ConstructMode);
            if (Ensemble.ConstructMode && Ensemble.ConstructMode ~= 4) && ~Ensemble.DivCrit, Ensemble.DivCrit = 2; end

        case 3
            % --------- OVERHAULED "Select the optimization function" -------
            if strcmpi(MODEFL,'classification')
            
                opts = [ ...
                  'Entropy (vote-entropy) [label-free vote dispersion]|' ...
                  '1 − Double-fault (A) [reduces simultaneous errors]|' ...
                  '−Q statistic [error-correlation; more negative ==> more diversity]|' ...
                  '(1 − Fleiss'' kappa)/2 [agreement over correctness; lower agreement ==> higher diversity]|' ...
                  '−LoBag ED [bias–variance Error Diversity; lower ED ==> better]|' ...
                  'COMBINE performance WITH a diversity metric' ];
            
                idxdef = local_default_idx_class(Ensemble); % your helper
                idx = nk_input('Select the optimization function',0,'m',opts,1:6,idxdef);
            
                switch idx
                    % ---- DIVERSITY ONLY ----
                    case 1  % Entropy only
                        Ensemble.DivCrit            = 1;
                        Ensemble.DiversitySource    = 'entropy';
                        Ensemble.DiversityObjective = 'H';   % canonical: entropy
            
                    case 2  % Double-fault A (min A)
                        Ensemble.DivCrit            = 3;
                        Ensemble.DiversitySource    = 'kappaa';  % for plotting/logs if you like
                        Ensemble.DiversityObjective = 'A';       % canonical
            
                    case 3  % Yule's Q (min Q)
                        Ensemble.DivCrit            = 3;
                        Ensemble.DiversitySource    = 'kappaq';
                        Ensemble.DiversityObjective = 'Q';
            
                    case 4  % Fleiss' kappa (min κ)
                        Ensemble.DivCrit            = 3;
                        Ensemble.DiversitySource    = 'kappaf';
                        Ensemble.DiversityObjective = 'K';
            
                    case 5  % LoBag only
                        Ensemble.DivCrit            = 4;
                        Ensemble.DiversitySource    = 'lobag';
                        Ensemble.DiversityObjective = 'L';  % optional canonical for LoBag
            
                    % ---- COMBINE perf + diversity ----
                    case 6
                        Ensemble.DivCrit            = 2;
                        Ensemble.DiversityObjective = [];         % not used in COMBINE
                        % ask which diversity to combine with:
                        Ensemble.DiversitySource    = local_pick_divsrc_class(Ensemble.DiversitySource);
                end
            
            else
                % ---------------- REGRESSION ----------------
                opts = [ ...
                  'Use regression ambiguity (nk_RegAmbig) ONLY [dispersion around ensemble mean]|' ...
                  'COMBINE performance WITH regression ambiguity' ];
            
                idxdef = (strcmpi(Ensemble.DiversitySource,'regvar') && Ensemble.DivCrit==2) + 1; % 1 or 2
                idx = nk_input('Select the optimization function',0,'m',opts,1:2,idxdef);
            
                if idx==1
                    % Diversity ONLY (reg ambiguity)
                    Ensemble.DivCrit            = 5;
                    Ensemble.DiversitySource    = 'regvar';
                    Ensemble.DiversityObjective = 'R';     % canonical: regression ambiguity
                else
                    % Combine perf + reg ambiguity
                    Ensemble.DivCrit            = 2;
                    Ensemble.DiversitySource    = 'regvar';
                    Ensemble.DiversityObjective = [];
                end
            end

        case 4
            Ensemble.Metric = nk_input('Use predicted labels or algorithm scores for ensemble construction',0,'m', ...
                ['Predicted labels (Hard decision ensemble)|' ...
                 'Algorithm scores (Soft decision ensemble)'], 1:2, Ensemble.Metric);

        case 5
            Ensemble.MinNum = nk_input('Minimum number of classifiers to retain', 0,'e',Ensemble.MinNum);

        case 6
            if ~Ensemble.Weighting, Ensemble.Weighting=2; end
            Ensemble.Weighting = nk_input('Weight base hypotheses?',0,'m', ...
                'No weighting|Weight = 1 / resubstitution error',[0,1], Ensemble.Weighting);

        case 7
            Ensemble.Perc = nk_input('[%] of cross-subspace feature agreement',0,'e',Ensemble.Perc);

        case 8
            Ensemble.MinNum = nk_input('Minimum number of features to retain',0,'e',Ensemble.MinNum);

        % ---- shown only when DivCrit==2 (combine) ----
        case 9
            if strcmpi(MODEFL,'classification')
                Ensemble.DiversitySource = local_pick_divsrc_class(Ensemble.DiversitySource);
            else
                Ensemble.DiversitySource = 'regvar';
                msgbox('Regression: using variance/ambiguity (nk_RegAmbig) as diversity term.');
            end

        case 10
            slackPct = nk_input('Performance slack as PERCENT of criterion range (e.g., 0.5)',0,'e',100*Ensemble.PerfSlackPct);
            Ensemble.PerfSlackPct = max(0, slackPct)/100;

        case 11
            Ensemble.EntropyWeight = nk_input('Weighted-rank diversity weight (w_E)',0,'e',Ensemble.EntropyWeight);

        case 12
            Ensemble.SizePenalty = nk_input('Size penalty λ (0=off)',0,'e',Ensemble.SizePenalty);

        case 13
            Ensemble.Patience = nk_input('Early-stop patience (steps, 0=off)',0,'e',Ensemble.Patience);

        case 14
            txt = local_regularizer_help(SVM, Ensemble);
            try
                helpdlg(txt, 'Regularizer guidance');
            catch
                fprintf('\n%s\n', txt);
            end
        case 15  % ε for DiversityMax
            Ensemble.DiversityEps = nk_input('Set diversity epsilon ε (tolerance)',0,'e',Ensemble.DiversityEps);
        case 16  % quick help
            txt = sprintf([ ...
                'Diversity-only knobs (nk_DiversityMax)\n' ...
                '-------------------------------------------------\n' ...
                'ε (DiversityEps):\n' ...
                '  • Acceptance tolerance (minimization): use strict cand < best − ε after MinNum.\n' ...
                '  • Typical: 0.01–0.05 relative to your diversity scale.\n' ...
                '\n' ...
                'γ (SizePenalty):\n' ...
                '  • Used in J(S) = D(S) + γ·log|S| (if you defined a size penalty in nk_DiversityMax).\n' ...
                '  • Larger |S| increases J; set γ=0 to disable size term.\n' ...
                '  • Typical: 0–0.1 relative to your diversity scale.\n' ...
                ]);
            try, helpdlg(txt,'Diversity-only guidance'); catch, fprintf('\n%s\n',txt); end
    end
end

% ----------------------- define objectives --------------------------------
if SubSpaceStrategy > 1
    switch Ensemble.DivCrit

        % -------- Entropy ONLY --------
        case 1
            Ensemble.DivFunc         = 'nk_Entropy';
            Ensemble.OptFunc         = 'DiversityMax';
            Ensemble.DiversitySource = 'entropy';
            Ensemble.DivStr          = 'vote-entropy';

        % -------- COMBINE perf + chosen diversity --------
        case 2
            if ~strcmpi(MODEFL,'classification')
                Ensemble.DiversitySource = 'regvar';
            end
            switch lower(Ensemble.DiversitySource)
                case 'entropy', Ensemble.DivFunc='nk_Entropy';        Ensemble.DivStr='prediction performance + vote-entropy';
                case 'kappaa',  Ensemble.DivFunc='nk_Diversity';      Ensemble.DivStr='prediction performance + (1 − double-fault A)';
                case 'kappaq',  Ensemble.DivFunc='nk_Diversity';      Ensemble.DivStr='prediction performance + (−Q)';
                case 'kappaf',  Ensemble.DivFunc='nk_DiversityKappa'; Ensemble.DivStr='prediction performance + (1 − Fleiss'' κ)/2';
                case 'lobag',   Ensemble.DivFunc='nk_Lobag';          Ensemble.DivStr='prediction performance + (−LoBag ED)';
                case 'regvar',  Ensemble.DivFunc='nk_RegAmbig';       Ensemble.DivStr='prediction performance + regression ambiguity';
                otherwise,      Ensemble.DivFunc='nk_Entropy';        Ensemble.DivStr='prediction performance + vote-entropy';
            end
            Ensemble.OptFunc = 'CVMax';
            [OptInline1, OptInline2]   = local_build_inlines(Ensemble.CompFunc, Ensemble.EntropRegMode);
            Ensemble.OptInlineFunc1    = OptInline1;
            Ensemble.OptInlineFunc2    = OptInline2;

        % -------- Kappa-family ONLY (driven by canonical DiversityObjective) --------
        case 3
            if ~isfield(Ensemble,'DiversityObjective') || isempty(Ensemble.DiversityObjective)
                Ensemble.DiversityObjective = 'Q';
            end
            switch upper(Ensemble.DiversityObjective)
                case 'A', Ensemble.DivFunc='nk_Diversity';      Ensemble.DivStr='Double-fault A (minimize A)';
                case 'Q', Ensemble.DivFunc='nk_Diversity';      Ensemble.DivStr='Yule''s Q (minimize Q)';
                case 'K', Ensemble.DivFunc='nk_DiversityKappa'; Ensemble.DivStr='Fleiss'' κ (minimize κ)';
                otherwise, Ensemble.DivFunc='nk_Diversity';     Ensemble.DivStr='Yule''s Q (minimize Q)';
            end
            Ensemble.OptFunc = 'DiversityMax';

        % -------- LoBag ONLY (classification) --------
        case 4
            Ensemble.DivFunc         = 'nk_Lobag';
            Ensemble.OptFunc         = 'DiversityMax';
            Ensemble.DiversitySource = 'lobag';
            Ensemble.DivStr          = 'LoBag error diversity (minimize ED)';

        % -------- Regression ambiguity ONLY --------
        case 5
            Ensemble.DivFunc         = 'nk_RegAmbig';
            Ensemble.OptFunc         = 'DiversityMax';
            Ensemble.DiversitySource = 'regvar';
            Ensemble.DivStr          = 'Regression ambiguity (dispersion around ensemble mean)';
    end
end

% ----------------------- map ConstructMode -> type ------------------------
switch Ensemble.ConstructMode
    case 0
        Ensemble.type = 0; % aggregate
    case 1  % Backward Elimination
        switch Ensemble.OptFunc
            case 'DiversityMax',  Ensemble.type = 1;
            case 'CVMax',         Ensemble.type = 2;
        end
    case 2  % Forward Selection
        switch Ensemble.OptFunc
            case 'DiversityMax',  Ensemble.type = 5;
            case 'CVMax',         Ensemble.type = 6;
        end
    case 4
        Ensemble.type = 9; % PFC
end

param.SubSpaceStrategy  = SubSpaceStrategy;
param.SubSpaceCrit      = SubSpaceCrit;
param.EnsembleStrategy  = Ensemble;

if exist('act','var') && act > 0
    param = nk_EnsembleStrategy_config(param, SVM, MODEFL, [], parentstr);
end

% ====================== helpers ===========================================
function Eout = local_overwrite_defaults(Def, In, MODEFL_)
    Eout = Def;
    fn = fieldnames(In);
    for i=1:numel(fn), Eout.(fn{i}) = In.(fn{i}); end
    if ~isfield(Eout,'EntropRegMode')   || isempty(Eout.EntropRegMode),   Eout.EntropRegMode   = 1; end
    if ~isfield(Eout,'CompFunc')        || isempty(Eout.CompFunc),        Eout.CompFunc        = 'max'; end
    if ~isfield(Eout,'DiversitySource') || isempty(Eout.DiversitySource)
        if strcmpi(MODEFL_,'classification'), Eout.DiversitySource='entropy'; else, Eout.DiversitySource='regvar'; end
    end
    if ~isfield(Eout,'PerfSlackPct')    || isempty(Eout.PerfSlackPct),    Eout.PerfSlackPct    = 0.0; end
    if ~isfield(Eout,'EntropyWeight')   || isempty(Eout.EntropyWeight),   Eout.EntropyWeight   = 2.0; end
    if ~isfield(Eout,'SizePenalty')     || isempty(Eout.SizePenalty),     Eout.SizePenalty     = 0.0; end
    if ~isfield(Eout,'Patience')        || isempty(Eout.Patience),        Eout.Patience        = 0;   end
    if ~isfield(Eout,'DiversityEps') || isempty(Eout.DiversityEps), Eout.DiversityEps = 0.02; end

end

function s = local_pretty_divsrc(src)
    switch lower(src)
        case 'entropy', s='vote-entropy';
        case 'kappaa',  s='1 − double-fault (A)';
        case 'kappaq',  s='−Q statistic';
        case 'kappaf',  s='(1 − Fleiss'' κ)/2';
        case 'lobag',   s='−LoBag ED';
        case 'regvar',  s='regression variance';
        otherwise,      s=src;
    end
end

function lab = local_divcrit_label(E)
    % Friendly label for the current selection (classification)
    switch E.DivCrit
        case 1, lab = 'Entropy';
        case 2, lab = ['Combine performance + ' local_pretty_divsrc(E.DiversitySource)];
        case 3, lab = [local_pretty_divsrc(E.DiversitySource) ' ONLY'];
        case 4, lab = 'LoBag ONLY';
        otherwise,   lab = '—';
    end
end

function lab = local_divcrit_label_reg(E)
    if E.DivCrit==2
        lab = 'Combine performance + regression ambiguity';
    else
        lab = 'Regression ambiguity ONLY';
    end
end

function idx = local_default_idx_class(E)
    % Map current (DivCrit, DiversitySource) to default menu index 1..6
    if E.DivCrit==1 && strcmpi(E.DiversitySource,'entropy'), idx=1; return; end
    if E.DivCrit==3 && strcmpi(E.DiversitySource,'kappaa'),  idx=2; return; end
    if E.DivCrit==3 && strcmpi(E.DiversitySource,'kappaq'),  idx=3; return; end
    if E.DivCrit==3 && strcmpi(E.DiversitySource,'kappaf'),  idx=4; return; end
    if E.DivCrit==4,                                         idx=5; return; end
    if E.DivCrit==2,                                         idx=6; return; end
    idx=6; % default to combine
end

function out = local_pick_divsrc_class(cur)
    opts = 'Vote-entropy|1 − Double-fault (A)|−Q statistic|(1 − Fleiss'' κ)/2|−LoBag ED';
    map  = {'entropy','kappaa','kappaq','kappaf','lobag'};
    def  = find(strcmpi(map,cur)); if isempty(def), def = 1; end
    idx  = nk_input('Pick the diversity term to combine with performance',0,'m',opts,1:5,def);
    out  = map{idx};
end

function [Opt1, Opt2] = local_build_inlines(compFunc, regMode)
    isMax = strcmpi(compFunc,'max');
    switch regMode
        case 1
            if isMax
                Opt1 = @(Perf,PerfCrit,Entropy,EntropyCrit) (Perf > PerfCrit) && (Entropy >= EntropyCrit);
                Opt2 = @(OrigPerf,OptPerf,OrigEntropy,OptEntropy) (OrigPerf >= OptPerf) || (OrigPerf == OptPerf && OrigEntropy > OptEntropy);
            else
                Opt1 = @(Perf,PerfCrit,Entropy,EntropyCrit) (Perf < PerfCrit) && (Entropy <= EntropyCrit);
                Opt2 = @(OrigPerf,OptPerf,OrigEntropy,OptEntropy) (OrigPerf <= OptPerf) || (OrigPerf == OptPerf && OrigEntropy < OptEntropy);
            end
        case 2
            if isMax
                Opt1 = @(Perf,PerfCrit,Entropy,EntropyCrit) ((Perf >= PerfCrit) && (Entropy >= EntropyCrit)) || ((Perf > PerfCrit) && (Entropy > EntropyCrit));
                Opt2 = @(OrigPerf,OptPerf,OrigEntropy,OptEntropy) (OrigPerf >= OptPerf) || (OrigEntropy > OptEntropy);
            else
                Opt1 = @(Perf,PerfCrit,Entropy,EntropyCrit) ((Perf <= PerfCrit) && (Entropy <= EntropyCrit)) || ((Perf < PerfCrit) && (Entropy < EntropyCrit));
                Opt2 = @(OrigPerf,OptPerf,OrigEntropy,OptEntropy) (OrigPerf <= OptPerf) || (OrigEntropy < OptEntropy);
            end
        otherwise
            [Opt1, Opt2] = local_build_inlines(compFunc, 1);
    end
end

end

function txt = local_regularizer_help(SVM, EnsStrat)
% Build a concise, metric-aware guidance string

[yl, ylb] = nk_GetScaleYAxisLabel(SVM);      % yl = [min max] of active criterion
span      = yl(2) - yl(1);
slackNow  = EnsStrat.PerfSlackPct * span;
slack05   = 0.005 * span;                    % 0.5% reference
lambda10  = max(eps, slackNow / 10);         % “10 learners ≈ one slack” rule

txt = sprintf([ ...
  'REGULARIZER GUIDANCE\n' ...
  'Metric: %s  |  Range: [%g .. %g]  (span = %g)\n' ...
  '------------------------------------------------------------------------\n' ...
  'Performance slack (PerfSlackPct)\n' ...
  '  • Purpose: ε-constraint on performance within each step; only candidates\n' ...
  '    within ±slack are considered, diversity breaks ties.\n' ...
  '  • Good starting range: 0.2%%–1.0%% of metric span; typical: 0.5%%.\n' ...
  '  • Current: %g %%  → absolute slack = %.6g\n' ...
  '  • Example at 0.5%%: absolute slack ≈ %.6g\n' ...
  '\n' ...
  'Diversity rank weight (EntropyWeight = w_E)\n' ...
  '  • Purpose: weight of diversity rank vs performance rank in the tie-break.\n' ...
  '  • Typical: 1.5–3.0. Lower if diversity overpowers perf; higher if many perf ties.\n' ...
  '  • Current: %g\n' ...
  '\n' ...
  'Size penalty (lambda)\n' ...
  '  • Purpose: prefer smaller/cheaper subsets. Acceptance checks a penalized\n' ...
  '    comparator: for maximization metrics, require perf_gain > λ·Δcost.\n' ...
  '  • Quick default: λ = slackAbs / 10  (≈ 10 learners equal one slack).\n' ...
  '  • With current slack: λ ≈ %.6g   (set EnsStrat.LearnerCost to weight cost).\n' ...
  '  • Typical manual range: (0.1–0.5)×slackAbs per learner.\n' ...
  '  • Current: %g\n' ...
  '\n' ...
  'Early-stop patience (Patience)\n' ...
  '  • Purpose: stop FS/BE after this many consecutive non-accepted steps.\n' ...
  '  • Typical: 1–2. Use 0 to disable.\n' ...
  '  • Current: %g\n' ...
  '\n' ...
  'Notes\n' ...
  '  • Candidate pool too big → reduce PerfSlackPct or increase λ slightly.\n' ...
  '  • Ensembles too small    → decrease λ or slightly increase PerfSlackPct.\n' ...
  '  • For pLR / unbounded-like metrics, span comes from nk_GetScaleYAxisLabel,\n' ...
  '    so the same %% slack rule remains metric-agnostic.\n' ...
  ], ...
  ylb, yl(1), yl(2), span, ...
  100*EnsStrat.PerfSlackPct, slackNow, slack05, ...
  EnsStrat.EntropyWeight, ...
  lambda10, EnsStrat.SizePenalty, ...
  EnsStrat.Patience);
end
