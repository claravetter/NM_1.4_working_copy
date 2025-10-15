#include "mex.h"
#include <vector>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
 #include <omp.h>
#endif

// [med, iq] = mexWnanMedianIQR(X, w)
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2) {
        mexErrMsgTxt("Usage: [med, iq] = mexWnanMedianIQR(X, w)");
    }
    if (nlhs > 2) {
        mexErrMsgTxt("Too many output arguments.");
    }
    // Input validation
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||
        !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) {
        mexErrMsgTxt("Both X and w must be real double arrays.");
    }

    // Dimensions
    const mwSize n = mxGetM(prhs[0]);
    const mwSize p = mxGetN(prhs[0]);
    double *X = mxGetPr(prhs[0]);
    double *w = mxGetPr(prhs[1]);
    const mwSize wn = mxGetNumberOfElements(prhs[1]);
    if (wn != n) {
        mexErrMsgTxt("Length of w must match number of rows of X.");
    }

    // Create outputs: 1 x p
    plhs[0] = mxCreateDoubleMatrix(1, p, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1, p, mxREAL);
    double *med = mxGetPr(plhs[0]);
    double *iq  = mxGetPr(plhs[1]);

    // Main loop over columns
    #pragma omp parallel for
    for (mwSize j = 0; j < p; ++j) {
        // Gather only non-NaN entries
        std::vector<std::pair<double,double>> vals;
        vals.reserve(n);
        double *col = X + j*n;
        for (mwSize i = 0; i < n; ++i) {
            double xv = col[i];
            if (!mxIsNaN(xv)) {
                vals.emplace_back(xv, w[i]);
            }
        }
        if (vals.empty()) {
            med[j] = mxGetNaN();
            iq[j]  = mxGetNaN();
            continue;
        }

        // Sort by value
        std::sort(vals.begin(), vals.end(),
                  [](const auto &a, const auto &b){ return a.first < b.first; });

        // Compute total weight
        double totalW = 0;
        for (auto &pr : vals) {
            totalW += pr.second;
        }

        // Prepare thresholds
        double t1 = 0.25 * totalW;
        double t2 = 0.50 * totalW;
        double t3 = 0.75 * totalW;

        // Sweep once to find quartile positions
        double cw = 0;
        double q1 = vals.back().first, q2 = q1, q3 = q1;
        for (auto &pr : vals) {
            cw += pr.second;
            if (cw >= t1 && q1 == vals.back().first) {
                q1 = pr.first;
            }
            if (cw >= t2 && q2 == vals.back().first) {
                q2 = pr.first;
            }
            if (cw >= t3) {
                q3 = pr.first;
                break;
            }
        }

        med[j] = q2;
        iq[j]  = q3 - q1;
    }
}
