function R = nk_BuildDiversityKernel(P, L, src, varargin)
% P: N x M predictions of M base learners
%    - for CLASSIFICATION sources: supply *hard labels per learner* (e.g., argmax or sign)
%    - for 'regvar' (REGRESSION): supply continuous predictions
% L: N x 1 labels (class ids / {-1,+1} / continuous), aligned to P rows
% src: same strings you already use in nm_diversity_score ('entropy','kappaa','kappaq','kappaf','lobag','regvar')
% task: 'classification' | 'regression'
% varargin: optional masks (logical Nx1) to select fitting rows, etc.

[~, M] = size(P);
R = zeros(M, M);

% (1) Pairwise build via your nm_diversity_score on {a,b}
for a = 1:M
  for b = a:M
    if a==b
      % Diagonal: self-diversity — set to agreement=1 (penalize redundancy) or 0.
      % Using 1 makes w'Rw discourage weight concentration even with one learner.
      R(a,a) = 1;
    else
      % Build 2-column subset and call your helper
      Dpair = nm_diversity_score(P(:,[a b]), L, src);
      % Your helper returns "higher = better (more diversity)" for all cases.
      % For a *penalty* kernel, larger entries should mean "more redundancy".
      % Therefore invert/monotone-map to a redundancy-like weight:
      %   redundancy = 1 - normalized_diversity
      % Most of your D are already in [0,1] after their mappings.
      redundancy = 1 - max(0, min(1, Dpair));
      R(a,b) = redundancy;
      R(b,a) = redundancy;
    end
  end
end

% (2) 'regvar' requires continuous predictions; others need hard labels:
% If caller passed soft probs/logits for classification, convert to hard votes before calling this function.
%   e.g., P = (Pprob > 0.5); or P = onehot2argmax(PprobK); done upstream.

% (3) Symmetrize, PSD-project, and scale so alpha is interpretable
R = (R + R.')/2;
[V,D] = eig(R); D = diag(D); D = max(D, 0); R = V*diag(D)*V';
s = norm(R, 2); if s>0, R = R / s; end
end

function D = nm_diversity_score(Psub, Lsub, src)
    % Psub: for classification should be hard labels (either {-1,+1} or class ids)
    K = numel(unique(Lsub(~isnan(Lsub))));
    switch lower(src)
        case 'entropy'
            % nk_Entropy remaps labels internally → works for binary & multi-class
            D = nk_Entropy(Psub, [], [], []);                       % higher = more diversity
        case 'kappaa'     % use double-fault A; invert to “higher=better”
            [A,~] = nk_Diversity(Psub, Lsub, [], []);
            if ~isfinite(A), D = 0; else, D = max(0, min(1, 1 - A)); end
        case 'kappaq'     % more negative Q is better → map [-1,1] → [0,1]
            [A,Q] = nk_Diversity(Psub, Lsub, [], []);
            if isfinite(Q)
                D = 0.5*(1 - max(-1, min(1, Q)));
            elseif isfinite(A)
                D = max(0, min(1, 1 - A));          % fallback to 1−A
            else
                D = 0;                               % neutral fallback
            end
        case 'kappaf'     % Fleiss' kappa; your impl returns 1-κ
            kdiv = nk_DiversityKappa(Psub, Lsub, [], []);
            if ~isfinite(kdiv), D = 0; else, D = max(0, min(1, 0.5*kdiv)); end
        case 'lobag'      % lower ED is better → we report as higher=better
            if K > 2
                D = -nk_LobagMulti_from_labels(Psub, Lsub);  % lower ED ⇒ higher diversity
            else
                D = -nk_Lobag(Psub, Lsub);
            end
        case 'regvar'     % regression ambiguity (var around ensemble)
            D = nk_RegAmbig(Psub, Lsub);
        otherwise
            D = nk_Entropy(Psub, [], [], []);
    end
end