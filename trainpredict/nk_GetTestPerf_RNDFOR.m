function [rs, ds] = nk_GetTestPerf_RNDFOR(~, tXtest, ~, md, ~, ~)
global MODEFL MODELDIR
    
    switch MODEFL
        case 'classification'
            %[rs, votes] = classRF_predict(tXtest,md); 
            model = md;

            %Get predictions and probabilities.
            if size(tXtest,1) == 1 && size(tXtest,2) == 1
                rs = double(model.predict(py.numpy.array(tXtest).reshape(int64(1), int64(1))).data);
                votes = double(model.predict_proba(py.numpy.array(tXtest).reshape(int64(1), int64(1))).data);
            elseif size(tXtest, 2) == 1 
                rs = double(model.predict(py.numpy.array(tXtest).reshape(int64(-1), int64(1))).data);
                votes = double(model.predict_proba(py.numpy.array(tXtest).reshape(int64(-1), int64(1))).data);
            elseif size(tXtest, 1) == 1
                rs = double(model.predict(py.numpy.array(tXtest).reshape(int64(1), int64(-1))).data);
                votes = double(model.predict_proba(py.numpy.array(tXtest).reshape(int64(1), int64(-1))).data);
            else
                rs = double(model.predict(tXtest).data);
                votes = double(model.predict_proba(tXtest).data);
            end

%             results_file = pyrunfile('cv_py_classRF_predict.py', ...
%                 'results_file' , model_name = md, test_feat =tXtest, ...
%                 rootdir = MODELDIR); 
%             results = load(char(results_file));
%             rs = results.predictions;
%             votes = results.probabilities;
            ds = votes(:,2)./sum(votes,2);
        case 'regression'
            %rs = regRF_predict(tXtest,md); ds=rs;
            model = md; 
            %Get predictions. 
            if size(tXtest,1) == 1 && size(tXtest,2) == 1
                rs = double(model.predict(py.numpy.array(tXtest).reshape(int64(1), int64(1))).data);
            elseif size(tXtest, 2) == 1 
                rs = double(model.predict(py.numpy.array(tXtest).reshape(int64(-1), int64(1))).data);
            elseif size(tXtest, 1) == 1
                rs = double(model.predict(py.numpy.array(tXtest).reshape(int64(1), int64(-1))).data);
            else
                rs = double(model.predict(double(tXtest)).data);
            end
            ds = rs;
    end
end
