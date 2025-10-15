% =========================================================================
% =                        VISUALIZATION PLOT                             =
% =========================================================================
function handles = display_visual(handles)
global st
st.ParentFig    = handles.figure1;
varind          = handles.curmodal;
measind         = handles.selVisMeas.Value;
meas            = handles.selVisMeas.String;
load_selPager(handles)
pageind         = handles.selPager.Value;
page            = handles.selPager.String;
pagemanual      = handles.txtPager.String;
sortfl          = handles.popupSortFeat.Value;
filterfl        = handles.tglVisMeas2.Value;
filterthr       = handles.txtThrVisMeas2.String;

handles.spiderPlotButton.Visible = "on";

axes(handles.axes33); cla; hold on
set(handles.axes33,'TickLabelInterpreter','none')

CorrMatStr = {'Correlation matrix', 'Correlation matrix (P value)','Correlation matrix (P value, FDR)'};
NetworkCorrMatStr = {'Network plot correlation matrix', 'Network plot correlation matrix (P value)', 'Network plot correlation matrix (P value, FDR)'};

% figure handle
hFig = handles.figure1;

% read colors (prefer handles, else appdata, else defaults)
if isfield(handles,'posBarColor') && ~isempty(handles.posBarColor)
    posColor = handles.posBarColor;
else
    posColor = getappdata(hFig,'posBarColor');
    if isempty(posColor), posColor = [0.85 0.15 0.15]; setappdata(hFig,'posBarColor',posColor); end
end
if isfield(handles,'negBarColor') && ~isempty(handles.negBarColor)
    negColor = handles.negBarColor;
else
    negColor = getappdata(hFig,'negBarColor');
    if isempty(negColor), negColor = [0.15 0.35 0.85]; setappdata(hFig,'negBarColor',negColor); end
end

v = handles.visdata{varind,handles.curlabel};
switch v.params.visflag
    case {1,2}
        featind = 1:v.params.nfeats;
    otherwise
        try
            if ~isempty(pagemanual) 
                featind = eval(pagemanual);
            else
                featind = eval(page{pageind});
            end
        catch
            featind = 1:v.params.nfeats;
        end
end

x = 0.5: numel(featind);
curclass = get(handles.popupmenu1,'Value');

if strcmp(handles.popupmenu1.String{curclass},'Multi-group classifier')
    multiflag = true;
else
    multiflag = false;
end

if measind>numel(meas)
    measind=numel(meas);
    handles.selVisMeas.Value=measind;
end

if strcmp(handles.selComponent.Visible,'on') && strcmp(handles.selComponent.Enable,'on')
    selC = handles.selComponent.Value-1;
else
    selC = 1;
end
if ~selC, selC=1; end

switch meas{measind} 
    case 'Model P value histogram'
        fl = 'off';
        fl2 = 'off'; 
        flhist = 'on';
    otherwise
        if isfield(v,'PermModel_Crit_Global_Multi') && multiflag
            fl = 'off';
        else
            fl = 'on';
        end
        flhist = 'on';
        if any(strcmp(meas{measind}, CorrMatStr))
            fl2 = 'on';
            load_selVisMeas2(handles)
        else
            if any(strcmp(meas{measind}, NetworkCorrMatStr))
                fl2 = 'on';
                load_selVisMeas2(handles)
            else
                fl2 = 'off';
            end
        end
end

switch v.params.visflag
    case 1
        handles.selPager.Enable = "off";
        handles.txtPager.Enable = "off";
        handles.tglSortFeat.Enable = "off";
        handles.cmdExportFeats.Enable = "off";
        handles.selVisMeas2.Enable = "off";
        handles.txtThrVisMeas2.Enable = "off";
        handles.tglVisMeas2.Enable = "off";
        handles.txtAlterFeats.Enable = "off";
        handles.cmdExportFeatFig.Enable = "off";
        handles.spiderPlotButton.Visible = "on";
        handles.btnPosColor.Visible = "off";
        handles.btnNegColor.Visible = "off";
    otherwise
        handles.selPager.Enable = fl;
        handles.txtPager.Enable = fl;
        handles.tglSortFeat.Enable = fl;
        handles.cmdExportFeats.Enable = fl;
        handles.selVisMeas2.Enable = fl2;
        handles.txtThrVisMeas2.Enable = fl2;
        handles.tglVisMeas2.Enable = fl2;
        handles.txtAlterFeats.Enable = fl;
        handles.btnPosColor.Visible = fl;
        handles.btnNegColor.Visible = fl;
        handles.cmdExportFeatFig.Enable = flhist;
        vlineval = [];
        if strcmp(fl,'on')
            if ~isempty(pagemanual) 
                handles.selPager.Enable = 'off';
            else
                handles.selPager.Enable = 'on';
            end
        end
        handles.spiderPlotButton.Visible = "off";
end

switch meas{measind}
    
    case 'Feature weights [Overall Mean (StErr)]'
        [y, se, miny, maxy, vlineval] = MEAN(v, curclass, multiflag);
    case 'Feature weights [CV2 Mean (StErr)]'
        [y, se, miny, maxy, vlineval] = MEAN_CV2(v, curclass, multiflag);
    case {'CVR of feature weights [Overall Mean]', 'CVR (SD-based) of feature weights [Overall Mean]', 'CVR (SEM-based) of feature weights [Overall Mean]'}
        [y, miny, maxy, vlineval] = CVRatio(v, curclass, multiflag);
        if miny>0, miny=0; end
    case {'CVR of feature weights [CV2 Mean]', 'CVR (SD-based) of feature weights [CV2 Mean]', 'CVR (SEM-based) of feature weights [CV2 Mean]'}
        [y, miny, maxy, vlineval] = CVRatio_CV2(v, curclass, multiflag);
        if miny>0, miny=0; end
    case 'Feature selection probability [Overall Mean]'
        [y, miny, maxy, vlineval] = FeatProb(v, curclass, multiflag);
    case 'Probability of feature reliability (95%-CI) [CV2 Mean]'
        [y, miny, maxy, vlineval] = Prob_CV2(v, curclass, multiflag);
    case 'Sign-based consistency'
        [y, miny, maxy, vlineval] = SignBased_CV2(v, curclass, multiflag);
        if miny>0, miny=0; end
    case 'Sign-based consistency (Z score)'
        [y, miny, maxy, vlineval] = SignBased_CV2_z(v, curclass, multiflag);
        if miny>0, miny=0; end
    case 'Sign-based consistency -log10(P value)'
        [y, miny, maxy, vlineval] = SignBased_CV2_p_uncorr(v, curclass, multiflag);
        if miny>0, miny=0; end
    case 'Sign-based consistency -log10(P value, FDR)'
        [y, miny, maxy, vlineval] = SignBased_CV2_p_fdr(v, curclass, multiflag);
        if miny>0, miny=0; end
    case 'Spearman correlation [CV2 Mean]'
        [y, miny, maxy, vlineval] = Spearman_CV2(v, curclass, multiflag);
    case 'Pearson correlation [CV2 Mean]'
        [y, miny, maxy, vlineval] = Pearson_CV2(v, curclass, multiflag);
    case 'Spearman correlation -log10(P value) [CV2 Mean]'
        [y, se, miny, maxy, vlineval] = Spearman_CV2_p_uncorr(v, curclass, multiflag);
    case 'Pearson correlation -log10(P value) [CV2 Mean]'
       [y, se, miny, maxy, vlineval] = Pearson_CV2_p_uncorr(v, curclass, multiflag);
    case 'Spearman correlation -log10(P value, FDR) [CV2 Mean]'
       [y, miny, maxy, vlineval] = Spearman_CV2_p_fdr(v, curclass, multiflag);
    case 'Pearson correlation -log10(P value, FDR) [CV2 Mean]'
       [y, miny, maxy, vlineval] = Pearson_CV2_p_fdr(v, curclass, multiflag);
    case 'Permutation-based Z Score [CV2 Mean]'
       [y, miny, maxy, vlineval] = PermZ_CV2(v, curclass, multiflag);
    case 'Permutation-based -log10(P value) [CV2 Mean]'
       [y, miny, maxy, vlineval] = PermProb_CV2(v, curclass, multiflag);
    case 'Permutation-based -log10(P value, FDR) [CV2 Mean]'
       [y, miny, maxy, vlineval] = PermProb_CV2_FDR(v, curclass, multiflag);
    case 'Analytical -log10(P Value) for Linear SVM [CV2 Mean]'
       [y, miny, maxy, vlineval] = Analytical_p(v, curclass, multiflag);
    case 'Analytical -log10(P Value, FDR) for Linear SVM [CV2 Mean]'
        [y, miny, maxy, vlineval] = Analyitcal_p_fdr(v, curclass, multiflag);
    case 'Correlation matrix'
        y = v.CorrMat_CV2{curclass};
        miny = min(y); maxy = max(y);
    case 'Correlation matrix (P value)'
        y = v.CorrMat_CV2_p_uncorr{curclass};
        miny = min(y); maxy = max(y);
    case 'Correlation matrix (P value, FDR)'
        y = v.CorrMat_CV2_p_fdr{curclass};
        miny = min(y); maxy = max(y);
    case 'Network plot correlation matrix'
        y = v.CorrMat_CV2{curclass};
        miny = min(y); maxy = max(y);
    case 'Network plot correlation matrix (P value)'
        y = v.CorrMat_CV2_p_uncorr{curclass};
        miny = min(y); maxy = max(y);
    case 'Network plot correlation matrix (P value, FDR)'
        y = v.CorrMat_CV2_p_fdr{curclass};
        miny = min(y); maxy = max(y);
    case {'Model P value histogram', 'Model P value histogram [Multi-group]'}
        if multiflag
            y = nm_nanmean(v.PermModel_Crit_Global); 
            vp = nm_nanmean(v.ObsModel_Eval_Global);
            ve = nm_nanmean(v.PermModel_Eval_Global);
        else         
            y = v.PermModel_Crit_Global; 
            vp = v.ObsModel_Eval_Global;
            ve = v.PermModel_Eval_Global;
        end
        perms = length(v.PermModel_Eval_Global(1,:));
        cl_face = rgb('SkyBlue');
        cl_edge = rgb('DeepSkyBlue');

    otherwise
        if contains(meas{measind},'Generalization analysis')
            % Define the regular expression pattern
            pattern = '\[#(\d+)\]';

            % Use regular expression to find matches in the string
            match = regexp(meas{measind}, pattern, 'tokens', 'once');

            % Check if a match is found
            if ~isempty(match)
                % Convert the matched string to a number
                LabelNum = str2double(match{1});
                y = v.ExtraL(LabelNum).VCV2MPERM_EVALFUNC_GLOBAL( curclass,:); 
                vp = v.ExtraL(LabelNum).VCV2MORIG_EVALFUNC_GLOBAL(curclass);
                ve = v.ExtraL(LabelNum).VCV2MPERM_GLOBAL(curclass,:);
                meas{measind} = 'Model P value histogram';
                perms = length(y);
                cl_face = rgb('Khaki');
                cl_edge = rgb('DarkKhaki');
            end

        end
end
        
if multiflag && isfield(v,'PermModel_Crit_Global_Multi')
    if measind == 1
         y = v.PermModel_Crit_Global_Multi; 
         vp = v.ObsModel_Eval_Global_Multi;
         ve = v.PermModel_Eval_Global_Multi;
    else
         y = v.PermModel_Crit_Global_Multi_Bin(measind-1,:); 
         vp = v.ObsModel_Eval_Global_Multi_Bin(measind-1);
         ve = v.PermModel_Eval_Global_Multi_Bin(measind-1,:);
    end
    perms = length(v.PermModel_Eval_Global_Multi);
    meas{measind} = 'Model P value histogram';
end

if ~strcmp(meas{measind}, 'Model P value histogram')
    y(~isfinite(y))=0;
    if width(y)>1 
        % Multiple components used
        % y = y(:,handles.selComponent.Value);
    end
end

switch meas{measind}

    case CorrMatStr
         set(handles.pn3DView,'Visible','off'); set(handles.axes33,'Visible','on'); cla; hold on
         feats = v.params.features; 
         if ~isempty(handles.txtAlterFeats.String)
            altfeatsvar = handles.txtAlterFeats.String;
            feats = evalin('base',altfeatsvar);
         end
         if filterfl
             FltMetr = handles.selVisMeas2.String{handles.selVisMeas2.Value};
             switch FltMetr
                 case {'CVR of feature weights [Overall Mean]', 'CVR (SD-based) of feature weights [Overall Mean]', 'CVR (SEM-based) of feature weights [Overall Mean]'}
                     [ythr,~,~,vlineval] = CVRatio(v,curclass,multiflag);
                     if isempty(filterthr)
                         handles.txtThrVisMeas2.String='-2 2'; 
                         filterthr = vlineval;
                     else
                         filterthr = str2double(filterthr);
                     end
                 case {'CVR of feature weights [CV2 Mean]', 'CVR (SD-based) of feature weights [CV2 Mean]', 'CVR (SEM-based) of feature weights [CV2 Mean]'}
                     [ythr,~,~,vlineval] = CVRatio_CV2(v,curclass,multiflag);
                     if isempty(filterthr)
                         handles.txtThrVisMeas2.String='-2 2'; 
                         filterthr = vlineval;
                     else
                         filterthr = str2double(filterthr);
                     end
                 case 'Feature selection probability [Overall Mean]'
                     [ythr,~,~,vlineval] = FeatProb(v,curclass,multiflag);
                     if isempty(filterthr)
                         handles.txtThrVisMeas2.String='0.5'; 
                         filterthr = vlineval;
                     else
                         filterthr = str2double(filterthr);
                     end
                 case 'Probability of feature reliability (95%-CI) [CV2 Mean]'
                     [ythr,~,~,vlineval] = Prob_CV2(v,curclass,multiflag);
                     if isempty(filterthr)
                         handles.txtThrVisMeas2.String='-0.5 0.5'; 
                         filterthr = vlineval;
                     else
                         filterthr = str2double(filterthr);
                     end
                 case 'Sign-based consistency -log10(P value)'
                     [ythr,~,~,vlineval] = SignBased_CV2_p_uncorr(v,curclass,multiflag);
                     if isempty(filterthr)
                         handles.txtThrVisMeas2.String=num2str(-log10(0.05),'%1.2f'); 
                         filterthr = vlineval;
                     else
                         filterthr = str2double(filterthr);
                     end
                 case 'Sign-based consistency -log10(P value, FDR)'
                     [ythr,~,~,vlineval] = SignBased_CV2_p_fdr(v,curclass,multiflag);
                     if isempty(filterthr)
                         handles.txtThrVisMeas2.String=num2str(-log10(0.05),'%1.2f'); 
                         filterthr = vlineval;
                     else
                         filterthr = str2double(filterthr);
                     end
                 case 'Analytical -log10(P Value) for Linear SVM [CV2 Mean]'
                     [ythr,~,~,vlineval] = Analytical_p(v,curclass,multiflag);
                     if isempty(filterthr)
                         handles.txtThrVisMeas2.String=num2str(-log10(0.05),'%1.2f'); 
                         filterthr = vlineval;
                     else
                         filterthr = str2double(filterthr);
                     end
                 case 'Analytical -log10(P Value, FDR) for Linear SVM [CV2 Mean]'
                     [ythr,~,~,vlineval] = Analyitcal_p_fdr(v,curclass,multiflag);
                     if isempty(filterthr)
                         handles.txtThrVisMeas2.String=num2str(-log10(0.05),'%1.2f'); 
                         filterthr = vlineval;
                     else
                         filterthr = str2double(filterthr);
                     end
             end
            % New (filter on): sort by the chosen metric ythr
             sI = get_sort_index(ythr(:), sortfl);
             y = y(sI,sI);
             feats = feats(sI);
             y = y(featind,featind);
             feats = feats(featind);
             if numel(filterthr)>1
                Ix2 = (ythr >= min(filterthr) | ythr <= max(filterthr))';
             else 
                Ix2 = (abs(ythr)>= filterthr)'; 
             end
             Ix2 = Ix2(sI); Ix2 = Ix2(featind);
             Ix = any(y) & Ix2;
         else
             % New (filter on): sort by the chosen metric
             sI = get_sort_index(mean(y), sortfl);
             y = y(sI,sI);
             feats = feats(sI);
             y = y(featind,featind);
             feats = feats(featind);
             Ix = any(y);
         end
         y = y(Ix,Ix);
         feats = feats(Ix);
         nF = numel(feats);
         if nF<50, FS = 10; elseif nF<100, FS = 7; elseif nF < 150, FS = 5; else, FS = 0; end
         yy = y; 
         try
             if strcmp(meas{measind}, CorrMatStr{1})
                     yy(itril(size(yy))) = -1.1;
                     imagesc( yy ); cbar = colorbar('Visible','on');
                     clabel = 'Correlation coefficient';
                     handles.axes33.Colormap = customcolormap([0 0.48 0.99 1],[rgb('Red'); rgb('White'); rgb('Blue'); rgb('LightGray'); ]);
                     handles.axes33.CLim = [-1.1 1];
                     cbar.Limits = [-1 1];
             else
                     yy(itril(size(yy))) = 0;
                     imagesc( yy ); cbar = colorbar('Visible','on');
                     if strcmp(meas{measind}, CorrMatStr{3})
                         clabel = 'P value (FDR-corrected)';
                     else
                        clabel = 'P value (uncorrected)';
                     end
                     handles.axes33.Colormap =  customcolormap([0 0.25 0.50 0.75 .99 1],[rgb('Red'); rgb('Yellow'); rgb('Cyan'); rgb('Blue'); rgb('DarkViolet'); rgb('LightGray'); ]);
                     handles.axes33.CLim = [ -log10(0.05) max(yy(:))] ;
                     cbar.Limits = [-log10(0.05) handles.axes33.CLim(end)];
             end
             cbar.Label.String=clabel; cbar.Label.FontWeight='bold';

             if FS>0
                handles.axes33.XAxis.FontSize = FS; 
                handles.axes33.YAxis.FontSize = FS; 
                handles.axes33.XTick = 1:sum(Ix); handles.axes33.XTickLabel = feats; handles.axes33.XTickLabelRotation=45;
                handles.axes33.YTick = 1:sum(Ix); handles.axes33.YTickLabel = feats; 
                handles.axes33.XAxis.Label.FontSize = FS;
                handles.axes33.YAxis.Label.FontSize = FS;
             else
                handles.axes33.XTickMode = 'auto';
                handles.axes33.XLimMode = 'auto';
                handles.axes33.XTickLabelMode = 'auto';
                handles.axes33.XTickLabelRotation=0;
                handles.axes33.YTickMode = 'auto';
                handles.axes33.YLimMode = 'auto';
                handles.axes33.YTickLabelMode = 'auto';
                handles.axes33.XAxis.FontSize = 10;
                handles.axes33.YAxis.FontSize = 10;
                handles.axes33.XAxis.Label.FontSize = 10;
                handles.axes33.YAxis.Label.FontSize = 10;
             end

             handles.axes33.XLim = [0.5 sum(Ix)+0.5]; 
             handles.axes33.YLim = [0.5 sum(Ix)+0.5];
             handles.axes33.XAxis.Label.String = 'Features';
         catch ERR
             errordlg(sprintf('Matrix cannot be displayed!\n%s', ERR.message));
         end
         
    case 'Model P value histogram'

         n         = numel(y);
         sigma     = nm_nanstd(y);
         iqrValue  = prctile(y,75) - prctile(y,25);
         hs        = 0.9 * min(sigma, iqrValue/1.34) * n^(-1/5);   % Silverman’s rule
         binWidth  = 2 * iqrValue / n^(1/3);                       % Freedman–Diaconis
         handles.pn3DView.Visible = 'off'; 
         handles.axes33.Visible = 'on'; cla; 
         hold on;
         % grab the old position so we don’t clobber bottom & height
         pos = handles.axes33.Position;     
        
         % center horizontally & make half‑width
         pos(1) = (1 - 0.5)/2;    % = 0.25
         pos(3) = 0.5;
        
         % write it back
         handles.axes33.Position = pos;

         ah = histogram(handles.axes33, y, ...
                            'Normalization','probability', ...
                            'BinWidth', binWidth, ...
                            'EdgeColor','none', ...
                            'FaceColor',cl_face); 
         maxah= nm_nanmax(ah.Values); ylim([0 (maxah + maxah*0.2)]); 
         [f, xi] = ksdensity(y, ...
                                'Function','pdf', ...
                                'Bandwidth', hs, ...
                                'NumPoints', 200 );
         plot(handles.axes33, xi, ...
                scaledata(f,[],0, maxah), ...
                'Color', cl_edge, ...
                'LineWidth',2);
         handles.axes33.YTick = 0:maxah/10:(maxah+maxah*0.2);
         yticklabels(handles.axes33,'auto')
         [~,xlb]=nk_GetScaleYAxisLabel(handles.NM.analysis{handles.curranal}.params.TrainParam.SVM);
         rg = range(y)*0.15; xl = [ min(y)-rg max(y)+rg ]; if vp>=xl(2), xl(2) = vp + rg; elseif vp<=xl(1), xl(1) = vp +rg; end
         xlim(xl); 
         xlabel(['Optimization criterion: ' xlb]);
         ylabel('Probability');
         hold on;
         Pval = sum(ve)/size(ve,2);
         xp = [ vp vp ]; yp = [ 0 maxah + maxah*0.2 ];
         if Pval ~= 0
            Pvalstr = sprintf('OOT %s=%1.2f\nOOT Significance: P=%g', xlb, vp, Pval);
         else
            Pvalstr = sprintf('OOT %s=%1.2f\nOOT Significance: P<%g', xlb, vp, 1/perms);
         end
         if exist("vp_cv2","var")
            xp_cv2 = [ vp_cv2 vp_cv2 ];
            patch('Faces', [1 2 3 4], 'Vertices', ...
                [ [ vp_cv2-vp_ci_cv2(1) 0 ]; [ vp_cv2-vp_ci_cv2(1) maxah ] ; [vp_cv2+vp_ci_cv2(1) maxah] ; [vp_cv2+vp_ci_cv2(1) 0 ] ], ...
                'FaceColor', rgb('LightSalmon'), 'EdgeColor', 'none', 'FaceAlpha', 0.3 );
            line( xp_cv2, yp ,'LineWidth',1,'Color',rgb('LightSalmon') );
            Pvalstr_cv2 = sprintf('Mean CV2 %s=%1.2f\nMean CV2 Significance: P=%g', xlb, vp_cv2, ve_cv2);
         end
         line(xp, yp ,'LineWidth',2,'Color','r');
         if exist("vp_cv2","var")
             legend({'Binned null distribution', 'Fitted null distribution', sprintf('95%%-CI Mean %s',xlb), Pvalstr_cv2, Pvalstr }, 'Location', 'northwest');
         else
            legend({'Binned null distribution', 'Fitted null distribution', Pvalstr}, 'Location', 'northwest');
         end

    case NetworkCorrMatStr
         
         set(handles.pn3DView,'Visible','off'); set(handles.axes33,'Visible','on'); cla; hold on
         feats = v.params.features; 
         if ~isempty(handles.txtAlterFeats.String)
            altfeatsvar = handles.txtAlterFeats.String;
            feats = evalin('base',altfeatsvar);
         end
       
         if filterfl
             FltMetr = handles.selVisMeas2.String{handles.selVisMeas2.Value};
             switch FltMetr
                 case {'CVR of feature weights [Overall Mean]', 'CVR (SD-based) of feature weights [Overall Mean]', 'CVR (SEM-based) of feature weights [Overall Mean]'}
                     [ythr,~,~,vlineval] = CVRatio(v,curclass,multiflag);
                     if isempty(filterthr)
                         handles.txtThrVisMeas2.String='-2 2'; 
                         filterthr = vlineval;
                     else
                         filterthr = str2double(filterthr);
                     end
                 case {'CVR of feature weights [CV2 Mean]', 'CVR (SD-based) of feature weights [CV2 Mean]', 'CVR (SEM-based) of feature weights [CV2 Mean]'}
                     [ythr,~,~,vlineval] = CVRatio_CV2(v,curclass,multiflag);
                     if isempty(filterthr)
                         handles.txtThrVisMeas2.String='-2 2'; 
                         filterthr = vlineval;
                     else
                         filterthr = str2double(filterthr);
                     end
                 case 'Feature selection probability [Overall Mean]'
                     [ythr,~,~,vlineval] = FeatProb(v,curclass,multiflag);
                     if isempty(filterthr)
                         handles.txtThrVisMeas2.String='0.5'; 
                         filterthr = vlineval;
                     else
                         filterthr = str2double(filterthr);
                     end
                 case 'Probability of feature reliability (95%-CI) [CV2 Mean]'
                     [ythr,~,~,vlineval] = Prob_CV2(v,curclass,multiflag);
                     if isempty(filterthr)
                         handles.txtThrVisMeas2.String='-0.5 0.5'; 
                         filterthr = vlineval;
                     else
                         filterthr = str2double(filterthr);
                     end
                 case 'Sign-based consistency -log10(P value)'
                     [ythr,~,~,vlineval] = SignBased_CV2_p_uncorr(v,curclass,multiflag);
                     if isempty(filterthr)
                         handles.txtThrVisMeas2.String=num2str(-log10(0.05),'%1.2f'); 
                         filterthr = vlineval;
                     else
                         filterthr = str2double(filterthr);
                     end
                 case 'Sign-based consistency -log10(P value, FDR)'
                     [ythr,~,~,vlineval] = SignBased_CV2_p_fdr(v,curclass,multiflag);
                     if isempty(filterthr)
                         handles.txtThrVisMeas2.String=num2str(-log10(0.05),'%1.2f'); 
                         filterthr = vlineval;
                     else
                         filterthr = str2double(filterthr);
                     end
                 case 'Analytical -log10(P Value) for Linear SVM [CV2 Mean]'
                     [ythr,~,~,vlineval] = Analytical_p(v,curclass,multiflag);
                     if isempty(filterthr)
                         handles.txtThrVisMeas2.String=num2str(-log10(0.05),'%1.2f'); 
                         filterthr = vlineval;
                     else
                         filterthr = str2double(filterthr);
                     end
                 case 'Analytical -log10(P Value, FDR) for Linear SVM [CV2 Mean]'
                     [ythr,~,~,vlineval] = Analyitcal_p_fdr(v,curclass,multiflag);
                     if isempty(filterthr)
                         handles.txtThrVisMeas2.String=num2str(-log10(0.05),'%1.2f'); 
                         filterthr = vlineval;
                     else
                         filterthr = str2double(filterthr);
                     end
             end
             % New (filter on): sort by the chosen metric ythr
             sI = get_sort_index(ythr(:), sortfl);
             y = y(sI,sI);
             feats = feats(sI);
             y = y(featind,featind);
             feats = feats(featind);
             %y.diag
             if numel(filterthr)>1
                Ix2 = (ythr >= min(filterthr) | ythr <= max(filterthr))';
             else 
                Ix2 = (abs(ythr)>= filterthr)'; 
             end
             Ix2 = Ix2(sI); Ix2 = Ix2(featind);
             Ix = any(y) & Ix2;
         else
             % New (filter on): sort by the chosen metric 
             sI = get_sort_index(mean(y), sortfl);
             y = y(sI,sI);
             feats = feats(sI);
             y = y(featind,featind);
             feats = feats(featind);
             Ix = any(y);
         end
         y = y(Ix,Ix);
         feats = feats(Ix);
         nF = numel(feats);
         if nF<50, FS = 10; elseif nF<100, FS = 7; elseif nF < 150, FS = 5; else, FS = 0; end
         yy = y; 
                        
         try
             cla reset;
             switch meas{measind}
                case NetworkCorrMatStr{1}
                 
                    gy = graph(yy);
                    gy = rmedge(gy, 1:numnodes(gy), 1:numnodes(gy));
                    pgy = plot(gy);
                    pgy.NodeLabel = feats;
                    pgy.EdgeCData = gy.Edges.Weight;
                    pgy.EdgeCData(pgy.EdgeCData > 0) = 0.5;
                    pgy.EdgeCData(pgy.EdgeCData < 0) = -0.5;
                    pgy.EdgeColor = 'flat';
                    colormap gray;
                    gy.Edges.LWidths = 7*abs(gy.Edges.Weight);
                    pgy.LineWidth = gy.Edges.LWidths;
                    pgy.NodeColor = 'k';
                 case NetworkCorrMatStr{2}
                    gy = graph(yy);
                    gy = rmedge(gy, 1:numnodes(gy), 1:numnodes(gy));
                    %nonsigIdx = find(gy.Edges.Weight >= 0.05);
                    %gy = rmedge(gy, nonsigIdx);
                    pgy = plot(gy);
                    pgy.NodeLabel = feats;
                    pgy.EdgeCData = gy.Edges.Weight;
                    pgy.EdgeCData(pgy.EdgeCData >= 0.05) = 0.5;
                    pgy.EdgeCData(pgy.EdgeCData < 0.05) = -0.5;
                    pgy.EdgeAlpha = 0.1;
                    %pgy.EdgeAlpha(pgy.EdgeCData < 0.05) = 0.5;
                    pgy.EdgeColor = 'flat';
                    colormap gray;
                    gy.Edges.LWidths = 0.1*(max(gy.Edges.Weight)+1 -(abs(gy.Edges.Weight)));
                    pgy.LineWidth = gy.Edges.LWidths;
                    pgy.NodeColor = 'k';
                 case NetworkCorrMatStr{3}
                    yy = triu(yy)+triu(yy,1)';
                    gy = graph(upper(yy));
                    gy = rmedge(gy, 1:numnodes(gy), 1:numnodes(gy));
                    %nonsigIdx = find(gy.Edges.Weight >= 0.05);
                    %gy = rmedge(gy, nonsigIdx);
                    pgy = plot(gy);
                    pgy.NodeLabel = feats;
                    pgy.EdgeCData = gy.Edges.Weight;
                    pgy.EdgeCData(pgy.EdgeCData >= 0.05) = 0.5;
                    pgy.EdgeCData(pgy.EdgeCData < 0.05) = -0.5;
                    %pgy.EdgeAlpha = 0.1;
                    %pgy.EdgeAlpha(pgy.EdgeCData < 0.05) = 0.5;
                    pgy.EdgeColor = 'flat';
                    colormap gray;
                    gy.Edges.LWidths = 0.1*(max(gy.Edges.Weight)+1 -(abs(gy.Edges.Weight)));
                    pgy.LineWidth = gy.Edges.LWidths;
                    pgy.NodeColor = 'k';
             end
             %cbar.Label.String=clabel; cbar.Label.FontWeight='bold';


         catch ERR
             errordlg(sprintf('Network plot cannot be displayed!\n%s', ERR.message));
         end
         %handles.axes33.Colormap =  colormap(redgreencmap); 
         
         %handles.axes33.XLim = [min(pgy.XData)-1 max(pgy.XData)+1]; 
         %handles.axes33.YLim = [min(pgy.YData)-1 max(pgy.YData)+1];
        
    otherwise 

        ind = get_sort_index(y(:,selC), sortfl);     % sort by y according to drop-down
        y   = y(ind,selC);  y = y(featind);
        % enable/disable color pickers based on sign content
        btnPos = findobj(hFig,'Tag','btnPosColor');
        btnNeg = findobj(hFig,'Tag','btnNegColor');
        if ~isempty(btnPos), set(btnPos,'Enable', ternary(any(y>0),'on','off'), 'BackgroundColor', posColor); end
        if ~isempty(btnNeg), set(btnNeg,'Enable', ternary(any(y<0),'on','off'), 'BackgroundColor', negColor); end
        if exist('se','var'), se = se(ind); se = se(featind); end

        switch v.params.visflag
            case {0, 3, 4, 5}
                set(handles.pn3DView,'Visible','off'); set(handles.axes33,'Visible','on'); 
                cla
                colorbar('Visible','off')
                handles.axes33.XLimMode = 'auto';
                handles.axes33.XTickLabelMode = 'auto';
                handles.axes33.XTickLabelRotation=0;
                hold on
                % Build a significance mask using existing vlineval logic
                if ~isempty(vlineval)
                    y2 = y; 
                    if numel(vlineval)>1
                        y2(y>vlineval(1) & y<vlineval(2)) = NaN; 
                    else
                        y2(y < vlineval) = NaN; 
                    end
                    sigMask = isfinite(y2);   % true = highlighted bar
                else
                    sigMask = true(size(y));
                end
                
                if sortfl == 6 || sortfl == 7, absFlag = true; else, absFlag = false; end

                % Value-scaled, sign-aware colored bars (red=pos, blue=neg)
                % pass colors into plotter
                plot_scaled_barh(handles.axes33, x, y, ...
                    'SigMask', sigMask, 'absFlag', absFlag, ...
                    'PosColor', posColor, 'NegColor', negColor);

                if exist('se','var')
                    if absFlag
                        ydraw = abs(y);
                    else
                        ydraw = y;
                    end
                    h = herrorbar(ydraw, x, se, se,'ko');
                    set(h,'MarkerSize',0.001);
                    ylimoff = nm_nanmax(se);
                else 
                    ylimoff = 0;
                end

                if ~isempty(vlineval) 
                    for o=1:numel(vlineval)
                        xline(vlineval(o),'LineWidth',1.5, 'Color', 'red');
                    end
                end
                xlabel(meas{measind},'FontWeight','bold');
                ylabel('Features','FontWeight','bold');
                set(gca,'YTick',x);
                if ~isempty(handles.txtAlterFeats.String)
                    altfeatsvar = handles.txtAlterFeats.String;
                    tfeats = evalin('base',altfeatsvar);
                    if isempty(tfeats)
                        warndlg(sprintf('Could not find the alternative feature vector ''%s'' in the MATLAB workspace!', altfeatsvar))
                        feats = v.params.features;
                    elseif numel(tfeats)~= numel(v.params.features)
                        warndlg(sprintf('The alternative feature vector doesn not have same feature number (n=%g) as the original vector (n=%g)',numel(tfeats),numel(v.params.features))) 
                        feats = v.params.features;
                    elseif ~iscellstr(tfeats) && ~isstring(tfeats)
                        warndlg('The alternative feature vector must be a cell array of strings!'); 
                        feats = v.params.features;
                    else
                        feats = tfeats;
                    end
                else
                    feats = v.params.features;
                end
                
                feats = feats(ind); 
                feats = feats(featind);
                if ~isempty(vlineval)
                    Ix = isnan(y2);
                    featsI = strcat('\color{gray}',feats(Ix));
                    feats(Ix) = featsI;
                end
                if isstring(feats(1))
                    feats = cellfun(@(x) char(strjoin(x, ',')), feats, 'UniformOutput', false);
                end
                feats = regexprep(feats,'_', '\\_');
                handles.axes33.YTickLabel = feats;
                handles.axes33.TickLabelInterpreter = 'tex';
                handles.axes33.YLim = [ x(1)-0.5 x(end)+0.5 ];
                if any(~isfinite([miny maxy]))
                    miny = min(y(isfinite(y))); maxy = max(y(isfinite(y)));
                    warning(sprintf('non-numeric values detect in feature weights (%s)',meas{measind}))
                end

                % --- X-axis limits: include error bars and a small padding ---
                % When absFlag is true, start at 0 and go to max(|y| + se) + pad.
                % Otherwise, span [min(y - se), max(y + se)] with symmetric padding.
                % 1) Gather vectors (handle missing se, NaNs robustly)
                yv  = y(:);
                if exist('se','var') && ~isempty(se)
                    sev = se(:);
                else
                    sev = zeros(size(yv));
                end
                
                % 2) Compute limits
                % --- X-axis limits: include error bars and a small padding on BOTH sides ---
                if absFlag
                    % Absolute view: draw bars at |y| but allow a small negative left margin
                    % so error bars close to zero remain visible.
                    yabs       = abs(yv);
                    rightEdge  = max(yabs + sev, [], 'omitnan');  if ~isfinite(rightEdge), rightEdge = 0; end
                    leftExtent = min(yabs - sev, [], 'omitnan');  if ~isfinite(leftExtent), leftExtent = 0; end
                    % span computed from the range we need to show (may include negative leftExtent)
                    span = max(rightEdge - min(0,leftExtent), eps);
                    pad  = 0.02 * span;                           % ~2% padding
                    left = min(0, leftExtent) - pad;              % allow slight negative to show errorbar caps
                    right = rightEdge + pad;
                    handles.axes33.XLim = [left, right];
                else
                    % Signed view: include both tails with padding
                    left  = min(yv - sev, [], 'omitnan');  if ~isfinite(left),  left  = 0; end
                    right = max(yv + sev, [], 'omitnan');  if ~isfinite(right), right = 0; end
                    span  = max(right - left, eps);
                    pad   = 0.02 * span;
                    handles.axes33.XLim = [left - pad, right + pad];
                end

                handles.axes33.XTickMode='auto';
                if numel(ind) ~= numel(handles.visdata_table(handles.curmodal,handles.curlabel).tbl.ind)
                    act = 'create';
                else
                    act = 'reorder';
                end
                
                ax = handles.axes33;
                ax.Position(2) = .075;

                % 2) Read current inner‐pos & tick‐label insets
                pos = ax.Position;    % [x y w h] of the bar‐area
                ti  = ax.TightInset;  % [L B R T] margins for labels & ticks
                
                % 3) User parameters
                pad       = 0.01;     % 1% padding from container edges
                minInnerW = 0.6;      % never let bar‐area go below 60% of the container
                
                % 4) Decide the inner‐width (w)
                %    — at least minInnerW, never expand beyond current pos(3)
                w = max(minInnerW, pos(3));
                
                % 5) Center the combined box [labels + bars] in [0,1]:
                outerW = w + ti(1) + ti(3);                 % total needed width
                x      = (1 - outerW)/2 + ti(1);           % left‐edge of the bar‐area
                
                pos(3) = w;    % set width
                pos(1) = x;    % set left‐edge
                
                % 6) If the left‐hand labels still overflow, bump right
                if pos(1) < ti(1) + pad
                    pos(1) = ti(1) + pad;
                end
                
                % 7) **Final clamp on the right** so nothing exceeds container
                rightEdge = pos(1) + pos(3) + ti(3);
                if rightEdge > 1 - pad
                    % shrink width so that rightEdge == 1 - pad
                    pos(3) = (1 - pad) - ti(3) - pos(1);
                end
                
                % 8) Write it back
                ax.Position = pos;

                handles.visdata_table(handles.curmodal, handles.curlabel) = create_visdata_tables(v, handles.visdata_table(handles.curmodal, handles.curlabel), ind, act);

            case {1,2}
                % Neuroimaging data view!
                [~,~,ext] = fileparts(v.params.brainmask);
                switch ext
                    case '.nii'
                        handles.pn3DView.Visible='on'; handles.axes33.Visible='off'; 
                        nk_WriteVol(y,'temp',2,v.params.brainmask,v.params.badcoords, ...
                            handles.NM.datadescriptor{v.params.varind}.threshval, ...
                            char(handles.NM.datadescriptor{v.params.varind}.threshop));
                        if ~isfield(handles,'niiViewer') || isempty(handles.niiViewer)
                            % first time: create everything and keep the handle
                            handles.niiViewer = overlay_nifti_gui(handles.NM.defs.atlas_path, 'temp.nii', handles.pn3DView, 0.5);
                        else
                            % just update the stats image
                            handles.niiViewer.update('temp.nii');
                        end
                    case {'.mgh','.mgz'}
                    case '.gii'
                        handles.pn3DView.Visible='on'; handles.axes33.Visible='off'; handles.axes27.Visible='off'; handles.axes28.Visible='off';
                        s = GIIread(v.params.brainmask);
                        save(gifti(struct('vertices',double(s.vertices),'faces',double(s.faces),'cdata',y)),'temp.gii');
                        cat_surf_render('Disp','temp.gii');
                end
        end


        legend('off');

end

guidata(handles.figure1, handles);

function [y, miny, maxy, vlineval] = CVRatio(v, curclass, multiflag)

if iscell(v.CVRatio)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.CVRatio,[],2),2);
    else
        y = v.CVRatio{curclass}; 
    end
else
    y = v.CVRatio; 
end
vlineval = [-2 2];
miny = min(y); maxy = max(y);

function [y, miny, maxy, vlineval] = CVRatio_CV2(v, curclass, multiflag)

if iscell(v.CVRatio_CV2)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.CVRatio_CV2,[],2),2);
    else
        y = v.CVRatio_CV2{curclass};  
    end
else
    y = v.CVRatio_CV2;
end
vlineval = [-2 2];
miny = min(y); maxy = max(y);

function [y, miny, maxy, vlineval] = FeatProb(v, curclass, multiflag)

if iscell(v.FeatProb)
    if multiflag
        y = nm_nanmean(v.FeatProb{1},2);
    else
        y = v.FeatProb{1}(:,curclass);  
    end   
else
    y = v.FeatProb;
end
vlineval = 0.5;
miny = 0; maxy = max(y);

function [y, miny, maxy, vlineval] = Prob_CV2(v, curclass, multiflag)

if iscell(v.Prob_CV2)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.Prob_CV2,[],2),2);
    else
        y = v.Prob_CV2{curclass};  
    end
else
    y = v.Prob_CV2;
end
vlineval = [-0.5 0.5];
miny = -1; maxy = 1;

function [y, miny, maxy, vlineval] = SignBased_CV2(v, curclass, multiflag)

if iscell(v.SignBased_CV2)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.SignedBased_CV2,[],2),2);
    else
        y = v.SignBased_CV2{curclass};
    end
else
    y = v.SignBased_CV2;
end
vlineval= [];
miny = 0; maxy = 1;

function [y, miny, maxy, vlineval] = SignBased_CV2_z(v, curclass, multiflag)
if iscell(v.SignBased_CV2_z)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.SignBased_CV2_z,[],2),2);
    else
        y = v.SignBased_CV2_z{curclass};
    end
else
    y = v.SignBased_CV2_z;
end
vlineval= [-2 2];
miny = min(y(:)); maxy = max(y(:));

function [y, miny, maxy, vlineval] = SignBased_CV2_p_uncorr(v, curclass, multiflag)

if iscell(v.SignBased_CV2_p_uncorr)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.SignBased_CV2_p_uncorr,[],2),2);
    else
        y = v.SignBased_CV2_p_uncorr{curclass};
    end
else
    y = v.SignBased_CV2_p_uncorr;
end 
miny = 0; maxy = max(y(:));
vlineval = -log10(0.05);

function [y, miny, maxy, vlineval] = SignBased_CV2_p_fdr(v, curclass, multiflag)

if iscell(v.SignBased_CV2_p_fdr)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.SignBased_CV2_p_fdr,[],2),2);
    else
        y = v.SignBased_CV2_p_fdr{curclass};
    end
else
    y = v.SignBased_CV2_p_fdr;
end
miny = 0; maxy = max(y(:));
vlineval = -log10(0.05);

function [y, miny, maxy, vlineval] = Spearman_CV2(v, curclass, multiflag)

if iscell(v.Spearman_CV2)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.Spearman_CV2,[],2),2);
    else
        y = v.Spearman_CV2{curclass};
    end
else
    y = v.Spearman_CV2;
end
miny = min(y(:)); maxy = max(y(:));
vlineval= [];

function [y, miny, maxy, vlineval] = Pearson_CV2(v, curclass, multiflag)

if iscell(v.Pearson_CV2)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.Pearson_CV2,[],2),2);
    else
        y = v.Pearson_CV2{curclass};
    end
else
    y = v.Pearson_CV2;
end
miny = min(y(:)); maxy = max(y(:));
vlineval= [];

function [y, se, miny, maxy, vlineval] = Spearman_CV2_p_uncorr(v, curclass, multiflag)

if iscell(v.Spearman_CV2_p_uncorr)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.Spearman_CV2_p_uncorr,[],2),2);
        if isfield (v,'Spearman_CV2_p_uncorr_STD')
            se = nm_nanmean(sqrt(nk_cellcat(v.Spearman_CV2_p_uncorr_STD,[],2)),2);
        end
    else
        y = v.Spearman_CV2_p_uncorr{curclass}; 
        if isfield (v,'Spearman_CV2_p_uncorr_STD')
            se = sqrt(v.Spearman_CV2_p_uncorr_STD{curclass});
        end
    end
else
    y = v.Spearman_CV2_p_uncorr;
    if isfield (v,'Spearman_CV2_p_uncorr_STD')
        se = sqrt(v.Spearman_CV2_p_uncorr_STD);
    end
end
miny = 0; maxy = max(y(:));
vlineval = -log10(0.05);

function [y, se, miny, maxy, vlineval] = Pearson_CV2_p_uncorr(v, curclass, multiflag)

if iscell(v.Pearson_CV2_p_uncorr)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.Pearson_CV2_p_uncorr,[],2),2);
        if isfield (v,'Pearson_CV2_p_uncorr_STD')
            se = nm_nanmean(sqrt(nk_cellcat(v.Pearson_CV2_p_uncorr_STD,[],2)),2);
        end
    else
        y = v.Pearson_CV2_p_uncorr{curclass};  
        if isfield (v,'Pearson_CV2_p_uncorr_STD')
            se = sqrt(v.Pearson_CV2_p_uncorr_STD{curclass});
        end
    end
else
    y = v.Pearson_CV2_p_uncorr;
    if isfield (v,'Pearson_CV2_p_uncorr_STD')
        se = sqrt(v.Pearson_CV2_p_uncorr_STD);
    end
end
miny = 0; maxy = max(y(:));
vlineval = -log10(0.05);

function [y, miny, maxy, vlineval] = Spearman_CV2_p_fdr(v, curclass, multiflag)
if iscell(v.Spearman_CV2_p_fdr)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.Spearman_CV2_p_fdr,[],2),2);
    else
        y = v.Spearman_CV2_p_fdr{curclass}; 
    end
else
    y = v.Spearman_CV2_p_fdr;
end
miny = 0; maxy = max(y(:));
vlineval = -log10(0.05);

function [y, miny, maxy, vlineval] = Pearson_CV2_p_fdr(v, curclass, multiflag)
if iscell(v.Pearson_CV2_p_fdr)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.Pearson_CV2_p_fdr,[],2),2);
    else
        y = v.Pearson_CV2_p_fdr{curclass};  
    end
else
    y = v.Pearson_CV2_p_fdr;
end
miny = 0; maxy = max(y(:));
vlineval = -log10(0.05);

function [y, miny, maxy, vlineval] = PermZ_CV2(v, curclass, multiflag)
if iscell(v.PermZ_CV2)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.PermZ_CV2,[],2),2);
    else
        y = v.PermZ_CV2{curclass};  
    end
else
    y = v.PermZ_CV2;
end
miny = min(y(:)); maxy = max(y(:));
vlineval = [];

function [y, miny, maxy, vlineval] = PermProb_CV2(v, curclass, multiflag)
if iscell(v.PermProb_CV2)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.PermProb_CV2,[],2),2);
    else
        y = v.PermProb_CV2{curclass};  
    end
else
    y = v.PermProb_CV2;
end
miny = 0; maxy = max(y(:));
vlineval = -log10(0.05);

function [y, miny, maxy, vlineval] = PermProb_CV2_FDR(v, curclass, multiflag)
if iscell(v.PermProb_CV2_FDR)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.PermProb_CV2_FDR,[],2),2);
    else
        y = v.PermProb_CV2_FDR{curclass};  
    end
else
    y = v.PermProb_CV2_FDR;
end
miny = 0; maxy = max(y(:));
vlineval = -log10(0.05);

function [y, miny, maxy, vlineval] = Analytical_p(v, curclass, multiflag)
if iscell(v.Analytical_p)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.Analytical_p,[],2),2);
    else
        y = v.Analytical_p{curclass};  
    end
else
    y = v.Analytical_p;
end
miny = 0; maxy = max(y(:));
vlineval = -log10(0.05);

function [y, miny, maxy, vlineval] = Analyitcal_p_fdr(v, curclass, multiflag)
if iscell(v.Analyitcal_p_fdr)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.Analyitcal_p_fdr,[],2),2);
    else
        y = v.Analyitcal_p_fdr{curclass};  
    end
else
    y = v.Analyitcal_p_fdr;
end
miny = 0; maxy = max(y(:));
vlineval = -log10(0.05);

function [y, se, miny, maxy, vlineval] = MEAN(v, curclass, multiflag)
if iscell(v.MEAN)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.MEAN,[],2),2);
        se = nm_nanmean(nk_cellcat(v.SE,[],2),2);
    else
        y = v.MEAN{curclass}; se = v.SE{curclass}; 
    end
else
    y = v.MEAN; se = v.SE; 
end
miny = min(y(:)); maxy = max(y(:));
vlineval = [];

function [y, se, miny, maxy, vlineval] = MEAN_CV2(v, curclass, multiflag)
if iscell(v.MEAN_CV2)
    if multiflag
        y = nm_nanmean(nk_cellcat(v.MEAN_CV2,[],2),2);
        se = nm_nanmean(nk_cellcat(v.SE_CV2,[],2),2);
    else
        y = v.MEAN_CV2{curclass}; se = v.SE_CV2{curclass}; 
    end
else
    y = v.MEAN_CV2; se = v.SE_CV2; 
end
miny = min(y(:)); maxy = max(y(:));
vlineval = [];

% you keep doing:
%   feats = feats(ind);
%   feats = feats(featind);
% This helper builds `ind` so that feats(featind) shows exactly what you want.

function ind = get_sort_index(vec, mode, featind)
% vec   : column or row vector of weights
% mode  : 1..7 as specified
% featind : positions for cutout (e.g., 1:41). If empty, we assume 1:K where K = numel(vec)

    vec = vec(:);
    n   = numel(vec);
    if nargin < 3 || isempty(featind)
        featind = 1:n;
    end
    % clamp cutout to valid range and compute the largest position we must support
    featind = featind(featind>=1 & featind<=n);
    if isempty(featind), ind = (1:n).'; return; end
    Kmax = max(featind);

    % push non-finites to the end, preserve their relative order
    finiteMask = isfinite(vec);
    idxF = find(finiteMask);
    idxNF = find(~finiteMask);

    % --- base orders ---
    switch mode
        case 1 % original
            base = (1:n).';
            % selected set for “relevance” is N/A; just return
            ind = base;
            return

        case 2 % ascending by value (neg -> pos)
            [~, o] = sort(vec(idxF), 'ascend');     base = [idxF(o); idxNF];

        case 3 % descending by value (pos -> neg)
            [~, o] = sort(vec(idxF), 'descend');    base = [idxF(o); idxNF];

        case {4,6} % absolute ascending (low -> high magnitude)
            [~, o] = sort(abs(vec(idxF)), 'ascend'); base = [idxF(o); idxNF];

        case {5,7} % absolute descending (high -> low magnitude)
            [~, o] = sort(abs(vec(idxF)), 'descend'); base = [idxF(o); idxNF];

        otherwise
            base = (1:n).';
    end

    % --- choose the “relevant” Kmax items per the scenario semantics ---
    switch mode
        case 2 % ascending: we want extremes first (most negative & most positive),
               % but final DISPLAY must remain monotonically ascending.
               % Strategy: take ends toward center until Kmax, then sort that slice ascending.
            ends = interleave_ends(1:n);                % positions in 'base'
            pick_pos = ends(1:Kmax);
            pick_idx = base(pick_pos);                  % original-row indices
            % re-order the picked set to be truly ascending for display
            [~, o2] = sort(vec(pick_idx), 'ascend');
            front = pick_idx(o2);
            rest  = setdiff(base, front, 'stable');
            ind   = [front; rest];

        case 3 % descending: same logic, but final order descending
            ends = interleave_ends(1:n);
            pick_pos = ends(1:Kmax);
            pick_idx = base(pick_pos);
            [~, o2] = sort(vec(pick_idx), 'descend');
            front = pick_idx(o2);
            rest  = setdiff(base, front, 'stable');
            ind   = [front; rest];

        case {4,6} % abs ascending: base is low->high | we STILL want most relevant in cutout
                   % i.e., take from the HIGH end (largest magnitudes) but keep abs-ascending order
            pick_pos = (n-Kmax+1):n;                    % last Kmax positions in base
            pick_idx = base(pick_pos);
            % keep their base order (abs-ascending within the chosen high-magnitude slice)
            front = pick_idx(:);
            rest  = setdiff(base, front, 'stable');
            ind   = [front; rest];

        case {5,7} % abs descending: base already high->low; take first Kmax
            pick_pos = 1:Kmax;
            pick_idx = base(pick_pos);
            front = pick_idx(:);                        % already in desired order
            rest  = setdiff(base, front, 'stable');
            ind   = [front; rest];

        otherwise
            ind = base;
    end

% helper: returns [1, n, 2, n-1, 3, n-2, ...] for positions 1..n
function out = interleave_ends(k)
    i = 1; j = numel(k); t = zeros(j,1);
    c = 1;
    while i <= j
        t(c) = k(i); c=c+1; i=i+1;
        if i <= j
            t(c) = k(j); c=c+1; j=j-1;
        end
    end
    out = t;

function hb = plot_scaled_barh(ax, pos, y, varargin)
% plot_scaled_barh  Horizontal bar chart with per-bar color scaling by value.
%   Red/Blue colors are user-selectable; intensity scales with |y|.
%   'SigMask' dims non-significant bars to light gray.
%
%   hb = plot_scaled_barh(ax, pos, y, 'SigMask', mask, 'absFlag', tf, ...
%                         'PosColor', [r g b], 'NegColor', [r g b])

    p = inputParser;
    addParameter(p, 'SigMask', true(size(y)), @(m) islogical(m) && numel(m)==numel(y));
    addParameter(p, 'absFlag', false, @(x) isscalar(x) && (islogical(x)||isnumeric(x)));
    addParameter(p, 'PosColor', [0.85 0.15 0.15], @(c) isnumeric(c)&&numel(c)==3);
    addParameter(p, 'NegColor', [0.15 0.35 0.85], @(c) isnumeric(c)&&numel(c)==3);
    parse(p, varargin{:});
    sigMask  = p.Results.SigMask(:);
    absFlag  = logical(p.Results.absFlag);
    posColor = p.Results.PosColor(:)';   % row vec
    negColor = p.Results.NegColor(:)';

    y   = y(:);
    pos = pos(:);

    maxAbs = max(abs(y));
    if isempty(maxAbs) || maxAbs==0
        colors = repmat([0.85 0.85 0.85], numel(y), 1);
    else
        alpha  = abs(y) ./ maxAbs;           % 0..1
        white  = [1 1 1];
        colors = zeros(numel(y),3);
        posIdx = y >= 0;
        colors(posIdx,:)  = white + (posColor - white) .* alpha(posIdx);
        colors(~posIdx,:) = white + (negColor - white) .* alpha(~posIdx);
    end

    if any(~sigMask)
        colors(~sigMask,:) = repmat([0.85 0.85 0.85], sum(~sigMask), 1);
    end

    yDraw = y; if absFlag, yDraw = abs(y); end
    hb = barh(ax, pos, yDraw, 'BarWidth',0.9, ...
        'FaceColor','flat','EdgeColor','none','LineWidth',1);
    hb.CData = colors;


function out = ternary(cond, a, b)
% TERNARY  Conditional selection like a ? b : c
%   out = ternary(cond, a, b)
%   - If COND is a logical scalar: returns A if true, else B.
%   - If COND is a logical array: returns elementwise A/B.
%     A or B may be scalars and will be expanded to size(COND).

    if ~islogical(cond)
        error('ternary: cond must be logical.');
    end

    if isscalar(cond)
        if cond, out = a; else, out = b; end
        return;
    end

    % Array case
    sz = size(cond);
    if isscalar(a), a = repmat(a, sz); end
    if isscalar(b), b = repmat(b, sz); end
    if ~isequal(size(a), sz) || ~isequal(size(b), sz)
        error('ternary: sizes of A and B must match COND (or be scalar).');
    end

    out = b;
    out(cond) = a(cond);
