// ============================================================================
// cbl_sampler_shim.cpp
//
// Thin C-ABI wrapper that lets a Python callable drive CBL's
// `cbl::statistics::Sampler::sample_stretch_move`. Adapts a plain C function
// pointer into the `std::function<double(std::vector<double>&)>` that the
// CBL constructor requires.
//
// Compile:
//   g++ -std=c++17 -fPIC -shared -fopenmp \
//       -I$CBL_HOME/Headers -I$CBL_HOME/External/Eigen/eigen-3.4.0 \
//       -I$CBL_HOME/External/Cuba-4.2.1 \
//       -Wl,-rpath,$CBL_HOME -L$CBL_HOME \
//       cbl_sampler_shim.cpp -lCBL -o cbl_sampler_shim.so
//
// Exposed:
//   int run_stretch_move(int npar, int nwalkers, int nsteps,
//                        const double* start_flat,
//                        double (*loglik)(const double*, int),
//                        unsigned int seed, double aa,
//                        double* out_chain_flat,
//                        double* out_mean_acceptance)
// where:
//   - start_flat has length nwalkers*npar (row-major)
//   - out_chain_flat must have capacity nsteps*nwalkers*npar
// ============================================================================

#include "Sampler.h"
#include <vector>
#include <functional>
#include <numeric>

extern "C" {

// Python-callable function pointer signature
typedef double (*py_loglik_t)(const double*, int);

// Expose the protected m_acceptance member by deriving from cbl::statistics::Sampler
class PythonSampler : public cbl::statistics::Sampler {
public:
    PythonSampler(int npar, std::function<double(std::vector<double>&)> f)
        : cbl::statistics::Sampler(npar, f) {}

    double get_mean_acceptance() const {
        if (m_acceptance.empty()) return 0.0;
        double sum = std::accumulate(m_acceptance.begin(), m_acceptance.end(), 0.0);
        return sum / m_acceptance.size();
    }
};

int run_stretch_move(int npar,
                     int nwalkers,
                     int nsteps,
                     const double* start_flat,
                     py_loglik_t loglik,
                     unsigned int seed,
                     double aa,
                     double* out_chain_flat,
                     double* out_mean_acceptance)
{
    try {
        // Wrap the C function pointer in a std::function with the signature
        // CBL expects. Captures `loglik` by value.
        std::function<double(std::vector<double>&)> f =
            [loglik, npar](std::vector<double>& x) -> double {
                return loglik(x.data(), npar);
            };

        // Construct Sampler and run
        PythonSampler sampler(npar, f);

        // Unflatten starting positions
        std::vector<std::vector<double>> start(nwalkers, std::vector<double>(npar));
        for (int w = 0; w < nwalkers; ++w)
            for (int p = 0; p < npar; ++p)
                start[w][p] = start_flat[w * npar + p];

        // sample_stretch_move(chain_size, nwalkers, start, seed, aa, outputFile)
        // (sequential variant; parallel uses OpenMP and would deadlock on Python GIL)
        sampler.sample_stretch_move(nsteps, nwalkers, start, seed, aa,
                                    cbl::par::defaultString);

        // Per CBL header: m_chains[step][walker][par], accessor get_chain(par, chain, step)
        for (int s = 0; s < nsteps; ++s)
            for (int w = 0; w < nwalkers; ++w)
                for (int p = 0; p < npar; ++p)
                    out_chain_flat[(s * nwalkers + w) * npar + p] =
                        sampler.get_chain(p, w, s);

        // Output mean acceptance fraction
        if (out_mean_acceptance) {
            *out_mean_acceptance = sampler.get_mean_acceptance();
        }

        return 0;
    }
    catch (cbl::glob::Exception& e) {
        return 1;
    }
    catch (std::exception& e) {
        return 2;
    }
}

} // extern "C"
