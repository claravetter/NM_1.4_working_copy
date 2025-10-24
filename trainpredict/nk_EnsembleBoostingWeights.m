function w = nk_EnsembleBoostingWeights(H, y, opts)
% H: N x M (probabilities or scores depending on mode)
% y: N x 1 (regression or binary {0,1}/{-1,+1}); multiclass = loop OVR
mode = getfielddef(opts,'mode','ridge'); % 'ridge'|'logloss'|'expgrad'
switch lower(mode)
  case 'ridge'
    lambda = getfielddef(opts,'lambda',1e-3);
    w = nm_weights_ridge_simplex(H, y, lambda);
  case 'logloss'
    lambda = getfielddef(opts,'lambda',1e-3);
    w = nm_weights_logloss_simplex(H, y, lambda);
  case 'expgrad'
    w = nm_weights_expgrad(H, y, opts);
  otherwise
    error('Unknown mode.');
end
end

function w = nm_weights_ridge_simplex(H, y, lambda)
% H: N x M, y: N x 1 (or prob target), lambda>=0
M = size(H,2);
Q = 2*(H.'*H + lambda*eye(M));
c = -2*H.'*y;

Aeq = ones(1,M); beq = 1;
A = -eye(M); b = zeros(M,1);  % w >= 0

opts = optimoptions('quadprog','Display','off','Algorithm','interior-point-convex');
w = quadprog(Q,c,A,b,Aeq,beq,[],[],[],opts);
end

function w = nm_weights_logloss_simplex(Hprob, y, lambda)
% Hprob: N x M of P(y=1) from base learners; y in {0,1}
% softmax parametrization w = softmax(u)
M = size(Hprob,2);
u0 = zeros(M,1);
obj = @(u) f(u,Hprob,y,lambda);
opts = optimoptions('fminunc','Algorithm','quasi-newton','Display','off','SpecifyObjectiveGradient',true);
u = fminunc(obj,u0,opts);
w = softmax(u);

function [L,g] = f(u,H,y,lambda)
  w = softmax(u);                % Mx1
  p = H*w;                       % Nx1 weighted prob
  p = min(max(p,1e-7),1-1e-7);   % clamp
  L = -sum(y.*log(p) + (1-y).*log(1-p)) + lambda*(w.'*w);
  % gradient wrt w
  gw = H.'*( (p - y) ) + 2*lambda*w;   % Mx1
  % chain rule through softmax: J = diag(w) - w*w'
  G = (diag(w) - w*w.') * gw;
  g = G;                          % gradient wrt u
end
end

function w = softmax(u)
u = u - max(u);
w = exp(u); w = w/sum(w);
end

function w = nm_weights_expgrad(Hscore, ypm1, opts)
% Hscore: N x M margin-like scores (larger = more +1), ypm1 in {-1,+1}
% opts: .eta (step), .alpha (diversity), .T (iters)
[~,M] = size(Hscore);
eta   = getfielddef(opts,'eta', 0.5);
alpha = getfielddef(opts,'alpha', 0.0);
T     = getfielddef(opts,'T',    200);

% init on simplex
w = ones(M,1)/M;

% precompute R for diversity (e.g., corr of predictions)
if alpha>0
  R = corr(Hscore); R(isnan(R))=0;
else
  R = [];
end

for t=1:T
  f = Hscore*w;                         % Nx1
  e = exp(-ypm1.*f);                    % weights like AdaBoost
  % gradient wrt w_m
  g = -(Hscore.'*(ypm1.*e));            % Mx1
  if alpha>0, g = g + 2*alpha*(R*w); end
  % exponentiated gradient step + simplex projection
  w = w .* exp(-eta*g);
  s = sum(w); if s==0, w = ones(M,1)/M; else, w = w/s; end
  % early stopping (relative change)
  if t>5 && norm(g,1)/M < 1e-6, break; end
end
end

function v = getfielddef(s,f,def), if isfield(s,f), v=s.(f); else, v=def; end, end
