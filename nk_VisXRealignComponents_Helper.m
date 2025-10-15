function [I, I1, Tx, Psel, Rx, SRx, PAx, assignmentVec, signCorrections] = ...
    nk_VisXRealignComponents(I, I1, h, Tx, Wx, Psel, Rx, SRx, PAx, Fadd, il, inp, nM, ill)
% nk_VisXRealignComponents
% Helper extracted from nk_VisModelsC.m — GLOBAL (CROSS-MODALITY) REALIGNMENT
%
% Returns updated I / I1 containers and working variables exactly as in the inline block.
%
% Dependencies (same as your original block):
%   - nk_AlignCompAndSignCorrect
%   - nan_l2n_block
%   - nm_nansum
global FUSION

    isInter = FUSION.flag == 2;

    % inp.decompfl is a logical/boolean vector (1 x nM or nM x 1) indicating DR-modalities
    if isInter
        assert(isfield(inp,'decompfl') && numel(inp.decompfl) >= nM, ...
            'Intermediate fusion requires inp.decompfl (length nM).');
        modMask = inp.decompfl(:);   % 1 x nM logical
    else
        modMask = true(1, nM);                  % early fusion: all modalities active
    end

    actMods   = find(modMask);                  % indices of active (DR) modalities
    nM_active = numel(actMods);
    
    nonZeroMasks = cell(nM,1);
    %% ---------- 1) Prune empty/NaN columns modality-wise, collect shapes ----------
    for n = 1:nM
        if ~modMask(n)
            % Non-DR modality in intermediate fusion -> leave as-is (ignored downstream)
            nonZeroMasks{n} = false(1,0);
            continue;
        end
        nz               = any((Tx{n}~=0) & isfinite(Tx{n}), 1);
        nonZeroMasks{n}  = nz;
        Tx{n}            = Tx{n}(:, nz);
    
        % keep siblings pruned the same way if they exist
        if exist('Psel','var') && ~isempty(Psel) && ~isempty(Psel{n}), Psel{n} = Psel{n}(:, nz); end
        if exist('Rx','var')   && ~isempty(Rx)   && ~isempty(Rx{n}),   Rx{n}   = Rx{n}(:,   nz); end
        if exist('SRx','var')  && ~isempty(SRx)  && ~isempty(SRx{n}),  SRx{n}  = SRx{n}(:,  nz); end
        if exist('PAx','var')  && ~isempty(PAx)  && ~isempty(PAx{n}),  PAx{n}  = PAx{n}(:,  nz); end
    end

    if isInter

        for n=1:nM
            if ~modMask(n), continue; end
            %% ---------- 2) Build or reuse the GLOBAL reference cache ----------
            haveRef = isfield(I,'VCV1REF') && numel(I.VCV1REF) >= h && ~isempty(I.VCV1REF{h}{n});
            % Ensure the cached ref cell exists and is length nM (placeholders for inactive)
            if ~haveRef, I.VCV1REF{h} = cell(1, nM); end


            
        end
    else



        %% ---------- 2) Build or reuse the GLOBAL reference cache ----------
        haveRef = isfield(I,'VCV1REF') && numel(I.VCV1REF) >= h && ~isempty(I.VCV1REF{h});
    
        
        % Prepare active slices
        
    
        %% ---------- 3) One call: realign (+ possibly grow ref) in blk-diag space ----------
        if haveRef
            [Tx, ~, assignmentVec, corrPerComp, signCorrections, ~, refUpdated_cell] = ...
                nk_AlignCompAndSignCorrect(I.VCV1REF{h}, Tx, inp.simCorrThresh, inp.simCorrMethod);
    
            % Cache updated reference (per-modality cells)
            I.VCV1REF{h} = refUpdated_cell;
    
            % Unified component count across modalities (must match by construction)
            colsPerMod     = cellfun(@(R) size(R,2), refUpdated_cell(:)');
            assert(all(colsPerMod == colsPerMod(1)), 'Ref columns must be equal across modalities after unified alignment.');
            nRef_combined  = colsPerMod(1);
        else
            % First time ever: seed with current kept maps; identity alignment
            I.VCV1REF{h}  = Tx;
            colsPerMod    = cellfun(@(R) size(R,2), I.VCV1REF{h}(:)');
            nRef_combined = colsPerMod(1);
            assignmentVec = (1:nRef_combined).';
            signCorrections = ones(nRef_combined,1);
            corrPerComp   = ones(nRef_combined,1);
            fprintf('\n\t\t\tDefine reference space consisting of %g component(s).', colsPerMod(1));
        end
    
        %% ---------- Ensure I1 destination containers exist *now* ----------
        need_bootstrap = ( ~isfield(I1,'VCV1WPERMREF')   || numel(I1.VCV1WPERMREF)   < h || isempty(I1.VCV1WPERMREF{h}) ) || ...
                         ( ~isfield(I1,'VCV1WCORRREF')   || numel(I1.VCV1WCORRREF)   < h || isempty(I1.VCV1WCORRREF{h}) ) || ...
                         ( ~isfield(I1,'ModComp_L2n')    || numel(I1.ModComp_L2n)    < h || isempty(I1.ModComp_L2n{h}) )  || ...
                         ( ~isfield(I1,'ModAgg_L2nShare')|| numel(I1.ModAgg_L2nShare)< h || isempty(I1.ModAgg_L2nShare{h}) ) || ...
                         ( ~isfield(I1,'ModComp_L2nCube')|| numel(I1.ModComp_L2nCube)< h || isempty(I1.ModComp_L2nCube{h}) );
    
        if need_bootstrap
            I1.VCV1WPERMREF{h}    = nan(nRef_combined, ill);
            I1.VCV1WCORRREF{h}    = nan(nRef_combined, ill);
            I1.ModComp_L2n{h}     = nan(nRef_combined, ill);
            I1.ModAgg_L2nShare{h} = nan(nM, ill);
            I1.ModComp_L2nCube{h} = nan(nRef_combined, nM, ill);
        end
    
        %% ---------- Grow containers if ref got longer ----------
        if size(I1.VCV1WPERMREF{h},1) < nRef_combined, I1.VCV1WPERMREF{h}(end+1:nRef_combined,:) = NaN; end
        if size(I1.VCV1WCORRREF{h},1) < nRef_combined, I1.VCV1WCORRREF{h}(end+1:nRef_combined,:) = NaN; end
        if size(I1.ModComp_L2n{h},1) < nRef_combined,  I1.ModComp_L2n{h}(end+1:nRef_combined,:) = NaN;  end
        if nM>1
            [s1, s2, s3] = size(I1.ModComp_L2nCube{h});
            if s1 < nRef_combined, I1.ModComp_L2nCube{h} = cat(1, I1.ModComp_L2nCube{h}, nan(nRef_combined - s1, s2, s3)); end
            if s3 < ill, I1.ModComp_L2nCube{h} = cat(3, I1.ModComp_L2nCube{h}, nan(size(I1.ModComp_L2nCube{h},1), s2, ill - s3)); end
        end
    
        %% ---------- p-values and correlations (component-level, modality-agnostic) ----------
        p_src_sig = I1.VCV1WPERM{h}(Fadd, il(h)); % compressed (sig-only)
        mask_ref   = assignmentVec > 0 & assignmentVec <= numel(p_src_sig);
        idx_pruned = assignmentVec(mask_ref);     % indices in p_src_sig
        p_ref = nan(numel(assignmentVec),1);
        p_ref(mask_ref) = p_src_sig(idx_pruned);
        I1.VCV1WPERMREF{h}(1:nRef_combined, il(h)) = p_ref;
        c = corrPerComp(:);
        oneTol = 1e-8; idx_nearOne = (c >= 1-oneTol) & (c <= 1+oneTol); c(idx_nearOne) = NaN;
        I1.VCV1WCORRREF{h}(1:nRef_combined, il(h)) = c;
    
        %% ---------- 6) L2 magnitudes (three views) ----------
        if exist('Wx','var') && ~isempty(Wx)
            if exist('Wx','var') && ~isempty(Wx)
                % Source component count across modalities (allow unequal)
                nComp_src = 0;
                for mx = 1:nM
                    nComp_src = max(nComp_src, size(Tx{mx}, 2));
                end
    
                % 1) Per-modality L2 vectors in source order (pad with NaN)
                L2_src_mod = nan(nComp_src, nM);   % nComp_src × nM
                for mx = 1:nM
                    if ~isempty(Wx{mx})
                        Xw = Wx{mx}; 
                        if size(Xw,2) == numel(nonZeroMasks{mx})
                            Xw = Xw(:, nonZeroMasks{mx});  % keep siblings pruned like Tx
                        end
                        v = nan_l2n_block(Xw);     % length = size(Xw,2)
                        if ~isempty(v)
                            L2_src_mod(1:numel(v), mx) = v;
                        end
                    end
                end
    
                % 2) Combined source L2 per component (aggregate across modalities)
                L2_src_combined = nm_nansum(L2_src_mod, 2);   % [nComp_src × 1]
    
                % 3) Map to reference order (NO tail blanking for L2)
                L2_ref_combined = nan(nRef_combined,1);
                keepL2 = assignmentVec > 0 & assignmentVec <= numel(L2_src_combined);
                L2_ref_combined(keepL2) = L2_src_combined(assignmentVec(keepL2));
    
                % (a) Global per-component stats
                I1.ModComp_L2n{h}(1:nRef_combined, il(h)) = L2_ref_combined;
    
                if nM>1
                    % 4) Per-modality slices for cube and shares
                    L2_ref_mod = nan(nRef_combined, nM);
                    ks = assignmentVec > 0 & assignmentVec <= nComp_src;
                    for mx = 1:nM
                        vec = nan(nRef_combined,1);
                        vec(ks) = L2_src_mod(assignmentVec(ks), mx);
                        L2_ref_mod(:, mx) = vec;
                        I1.ModComp_L2nCube{h}(1:nRef_combined, mx, il(h)) = vec;
                    end
    
                    % 5) Aggregate shares (robust, per-mod totals → normalize; preserve all-NaN as NaN)
                    share_vec = nan(nM,1);
                    mod_sum   = nan(nM,1);
                    mod_has   = false(nM,1);
                    for mx = 1:nM
                        col = L2_ref_mod(:, mx);
                        mod_has(mx) = any(isfinite(col));
                        if mod_has(mx)
                            mod_sum(mx) = nm_nansum(col);
                        else
                            mod_sum(mx) = NaN;  % keep as NaN so this modality's share stays NaN
                        end
                    end
    
                    denom = nm_nansum(mod_sum(mod_has));  % sum across modalities that had any finite L2
                    if denom > 0 && isfinite(denom)
                        share_vec(mod_has) = mod_sum(mod_has) ./ denom;
    
                        % clamp tiny FP drift and guard against nonsense
                        eps1 = 1e-9;
                        share_vec(share_vec < 0    & abs(share_vec) < eps1) = 0;
                        share_vec(share_vec > 1    & (share_vec - 1) < eps1) = 1;
                        bad = share_vec < -eps1 | share_vec > 1+eps1 | ~isfinite(share_vec);
                        share_vec(bad) = NaN;
    
                        % optional: renormalize finite entries to sum to 1 exactly
                        sfin = sum(share_vec(isfinite(share_vec)));
                        if sfin > 0
                            share_vec(isfinite(share_vec)) = share_vec(isfinite(share_vec)) ./ sfin;
                        end
                    else
                        share_vec(:) = NaN;  % no energy this model → all NaN
                    end
                    I1.ModAgg_L2nShare{h}(:, il(h)) = share_vec;
                end
            end
        end
    end
end
