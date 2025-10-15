function L2n = nan_l2n_block(x)
% L2n per column, robust to NaNs/Infs.
% If a column has no finite entries, L2n = NaN for that column.

    if isempty(x)
        L2n = [];  % nothing to compute
        return
    end

    % Vector case -> single scalar
    if isvector(x)
        w = x(:);
        m = isfinite(w);
        if ~any(m)
            L2n = NaN;
            return
        end
        L2n = norm(w(m), 2) / sqrt(sum(m));
        return
    end

    % Matrix case -> column vector result
    m   = isfinite(x);              % logical mask of finite entries
    cnt = sum(m, 1);                % finite count per col
    L2n = nan(size(x,2), 1);        % default NaN for empty columns

    nz = cnt > 0;                   % columns with at least one finite entry
    if any(nz)
        % Zero-out non-finite entries (avoid NaNs in sums), then sum squares
        tmp = x(:, nz);
        % For speed with sparse/dense, avoid elementwise multiply with mask; assign zeros instead
        tmp(~isfinite(tmp)) = 0;
        ssq = sum(tmp.^2, 1); % 1 x nnz cols

        L2n(nz) = sqrt(ssq(:)) ./ sqrt(cnt(nz)).';
    end
end