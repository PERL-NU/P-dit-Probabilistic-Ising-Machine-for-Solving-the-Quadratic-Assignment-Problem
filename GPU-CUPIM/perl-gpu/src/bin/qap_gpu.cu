// TODO Final review of initializations (vs old version)

/* standard */
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>

/* external */
#include <argparse.hpp>

/* internal */
#include "logger.cuh"
#include "qap.cuh"

/** Conversion ratio from log10 to log2 */
#define EXP2_SCALING 1.44269504089

/** Floating point precision used on the device */
using FloatingPoint = float;

using Clock = std::chrono::high_resolution_clock;
using Moment = std::chrono::time_point<Clock>;

//======================================================================================
// Manual Configuration
//======================================================================================

/** Toggle between using beta sweep and passing in beta */
static const bool kDefaultDoBetaSweep = true;

/** Toggle between best-found (false) and three-step plateau checking methods */
static const bool kUseFullPlateauCheck = false;

/** Number of mega-iterations until switching to the fine-grained algorithm */
static const int32_t kGibbsPatience = INT32_MAX;

/** Number of mega-iterations before recomputing energy delta */
static const int32_t kDriftPatience = INT32_MAX;

/** Toggle checkpointing vs full trial randomization */
static const bool kRandomTrialInit = true;

/** Set number of beta indices to add to max beta index found by sweep */
static const int64_t kSweepCautionRight = 2;

/** Set number of beta indices to subtract from min beta index found by sweep */
static const int64_t kSweepCautionLeft = 0;

//======================================================================================
// Default Commandline Arguments
//======================================================================================

/** Default number of iterations in a mega-iteration when none is provided via `-i` */
static const uint32_t kDefaultNumInnerIter = 4;

/** Default limit on mega-iterations when none is provided via `-w` */
static const uint32_t kDefaultMegaIterLim = UINT32_MAX;

/** Default pool width when none is provided via `-r` */
static const uint32_t kDefaultPoolWidth = 16;

/** Default pool depth when none is provided via `-t` */
static const uint32_t kDefaultPoolDepth = 32;

/** Default maximum beta value when none is provided via `-b` */
static const FloatingPoint kDefaultBetaMax = exp2(-12);

/** Default divisor for subsequent betas when none is provided via `-m` */
static const FloatingPoint kDefaultBetaDivisor = 1.2;

/** Default lower proportion in the beta sweep when none is provided via `--al` */
static const FloatingPoint kDefaultBetaPropLo = 0.0;

/** Default lower proportion in the beta sweep when none is provided via `--ah` */
static const FloatingPoint kDefaultBetaPropHi = 0.4;

/** Default score threshld to halt when none is provided via `-g` */
static const int32_t kDefaultGoalHalt = 0;

/** Default score threshold to print solution when none is provided via `-G` */
static const int32_t kDefaultGoalPrint = 0;

//======================================================================================
// Argparser
//======================================================================================

// TODO Many of these could probably be commandline arguments

static const FloatingPoint kMinBetaToTry = -30;  // TODO Commandline arguments
static const FloatingPoint kMaxBetaToTry = 1;
static const uint32_t kNumBetaTrials = 5;
//static const FloatingPoint kMarginFactor = 0.05;  // TODO Why unused?
static const FloatingPoint kBetaStep = 1;
static const uint32_t kNumBetaStep = (kMaxBetaToTry - kMinBetaToTry) / kBetaStep + 1;

/** Call-time configuration */
struct AppArgs: public argparse::Args {

    std::string &input_file = arg("input-file",
        "input file"
    );

    bool &do_beta_sweep = arg("sweep",
        "perform beta sweep (automatic mode)"
    ).set_default(kDefaultDoBetaSweep);

    FloatingPoint &beta_max = kwarg("b",
        "set the beta of the largest beta replica. automatic mode is used by default. "
        "in automatic mode, a series of short trials at different betas are used to "
        "try and choose the best parameters"
    ).set_default(kDefaultBetaMax);

    FloatingPoint &beta_min = kwarg("beta-min",
        "min beta used when automatic mode disabled"
    ).set_default(kDefaultBetaMax);  // TODO Sensible default

    FloatingPoint &beta_prop_lo = kwarg("al",
        "lower beta proportion"
    ).set_default(kDefaultBetaPropLo);

    FloatingPoint &beta_prop_hi = kwarg("ah",
        "higher beta proportion"
    ).set_default(kDefaultBetaPropHi);

    int &goal_halt = kwarg("g",
        "set the goal objective value, program halts when reached"
    ).set_default(kDefaultGoalHalt);

    int &goal_print = kwarg("G",
        "set the goal objective value, prints solution when reached"
    ).set_default(kDefaultGoalPrint);

    int &num_inner_iter = kwarg("i",
        "set number of iterations between replica swaps, defaults to 4"
    ).set_default(kDefaultNumInnerIter);

    FloatingPoint &beta_divisor = kwarg("m",
        "set the multiplative difference in beta replicas"
    ).set_default(kDefaultBetaDivisor);

    std::string &outdir = kwarg("o",
        "set the output folder name"
    ).set_default("");

    uint32_t &mega_iter_lim = kwarg("w",
        "maximum number of mega-iterations to perform"
    ).set_default(kDefaultMegaIterLim);

    std::optional<uint32_t> &timeout = kwarg("s",
        "set the timeout in seconds, program halts when reached (if provided)"
    );

    uint32_t &pool_width = kwarg("pool-width",
        "width of the replica pool. beta values descend across width"
    ).set_default(kDefaultPoolWidth);

    uint32_t &pool_depth = kwarg("pool-depth",
        "depth of the replica pool. this is the number of replicas with a shared beta"
    ).set_default(kDefaultPoolDepth);

    bool &large_energies = kwarg("large-energies",
        "problem is at risk of overflowing int32"
    ).set_default(false);

    // Derived
    std::string logfile;
};

static void validate_args(const AppArgs &args) {
    // TODO See if anything needs validating
}

static void set_derived_args(AppArgs &args) {
    if (args.outdir.empty()) {
        args.outdir = std::filesystem::absolute(args.input_file + ".out");
    }
    args.outdir = std::filesystem::absolute(args.outdir);
    args.logfile = std::format("{}/log.txt", args.outdir);
    //args.beta_max *= EXP2_SCALING;  // account for use of exp2 instead of exp

    // TODO Automatic mode stuff
}

static void log_args(const AppArgs &args) {
    //logger::write(std::format("Input file: {}\n", args.input_file));
    //logger::write(std::format("Iterations between swaps {}\n", args.inner_iter));
    //logger::write(std::format("Number of Swaps {}\n", args.num_swap));
    //logger::write(std::format("Number of Replicas {}\n", args.num_replica));
    //logger::write(std::format("Number of Threads {}\n", args.num_thread));
    // TODO maximum Beta/automatic mode stuff
}

static inline int32_t ms_passed(const Moment start_time, const Moment end_time) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    ).count();
}

template <typename Z>
void reset_best_scores(qap::ReplicaPool<FloatingPoint, Z> &pool_d) {
    cudaMemset(pool_d.best_score_per_replica, 0x7f, pool_d.num_replica * sizeof(int32_t));
    *pool_d.best_score_host = INT32_MAX;
}

template <typename Z>
static int32_t run_annealing_loop(
    qap::ReplicaPool<FloatingPoint, Z> &annealer_d,
    const AppArgs &args,
    const uint32_t num_mega_iter,
    const bool automatic_mode,
    const Moment start_time,
    std::ofstream &sweep_file,
    const int32_t best_score_sweep
) {
    bool use_coarse = true;
    uint32_t num_unimproved_runs = 0;

    int32_t best_score[1] = {best_score_sweep};
    int32_t last_best_score = best_score_sweep;

    std::random_device dev;
    auto host_rng = std::mt19937(dev());
    for (int m = 0; m < num_mega_iter; ++m) {
        if (kDriftPatience != INT32_MAX && m % kDriftPatience == 0) {
            annealer_d.recompute_force_mats(32);
            annealer_d.template recompute_energies<32>();
        }
        num_unimproved_runs++;
        if (kGibbsPatience != INT32_MAX && num_unimproved_runs >= kGibbsPatience) {
            use_coarse = false;
        }

        for (int i = 0; i < args.num_inner_iter; ++i) {
             double z = std::generate_canonical<double, std::numeric_limits<double>::digits>(
                host_rng
            );
            if (use_coarse) {
                annealer_d.template perform_step_one_shot<1024>(UINT64_MAX * z);
            } else {
                annealer_d.template perform_step<1024>(UINT64_MAX * z);
            }
        }
        annealer_d.thunk_scores(best_score, nullptr, nullptr, nullptr);

        // Bookkeeping and logging for new best scores
        const bool found_better = (*best_score < last_best_score);
        if (found_better) {
            num_unimproved_runs = 0;
            Moment check_time = Clock::now();
            logger::write(std::format(
                "Best Energy Found: {} Time (s): {}\n",
                *best_score,
                ms_passed(start_time, check_time) / 1000.0
            ));

            // TODO Open/close on every new best slows hotter betas disprop, so would
            //      ideally just open once at top but low priority
            sweep_file.open(
                args.outdir + "/sweep.txt",
                std::ios::app
            );  // TODO See if can recover thread ID
            sweep_file
                << m << " "
                << ms_passed(start_time, check_time) << " "
                << *best_score << "\n"
            ;
            sweep_file.close();
            last_best_score = *best_score;
        }
        last_best_score = (last_best_score < *best_score) ? last_best_score : *best_score;  // TODO Shouldn't need, was paranoid

        // Exit condition: reached target score
        if (last_best_score <= args.goal_halt) {
            logger::write(std::format(
                "Halting at {} mega-iterations with score {}: hit target score {}\n",
                m + 1, last_best_score, args.goal_halt
            ));
            return last_best_score;
        }

        // Update annealing ladder
        if (!automatic_mode) {
            const uint32_t pool_depth = annealer_d.pool_depth;

            if (pool_depth <= 32) {
                annealer_d.template percolate_betas<32>();
            } else if (pool_depth <= 64) {
                annealer_d.template percolate_betas<64>();
            } else if (pool_depth <= 128) {
                annealer_d.template percolate_betas<128>();
            } else if (pool_depth <= 256) {
                annealer_d.template percolate_betas<256>();
            } else if (pool_depth <= 512) {
                annealer_d.template percolate_betas<512>();
            } else if (pool_depth <= 1024) {
                annealer_d.template percolate_betas<1024>();
            } else {
                throw std::runtime_error("Maximum depth (1024) exceeded");
            }
        }

        // Exit condition: timed out
        if (!args.timeout.has_value()) {
            continue;
        }
        Moment check_time = Clock::now();
        if (ms_passed(start_time, check_time) >= *args.timeout) {
            logger::write(std::format(
                "Halting at {} mega-iterations with score {}: timed out\n",
                m + 1, last_best_score
            ));
            return last_best_score;
        }
    }

    if (!automatic_mode) {
        logger::write(std::format(
            "Completed {} mega-iterations and achieved score {}\n",
            num_mega_iter, last_best_score
        ));
    }
    return last_best_score;
}

template <typename T>
static void sigmoid_bookends_alt(
    T *sigmoid,
    const int sigmoid_len,
    uint32_t *bookends, // return variable
    const int window_len=3,
    const double thresh_sigma=1.0  // TODO CLI parameter
) {
    auto windowed_mean = [&](int start, int count) {
        double sum = 0.0;
        for (int i = start; i < start + count; ++i) {
            sum += sigmoid[i];
        }
        return sum / count;
    };
    auto windowed_sdev = [&](int start, int count, double mean) {
        double sum = 0.0;
        for (int i = start; i < start + count; ++i) {
            double x = sigmoid[i] - mean;
            sum += x * x;
        }
        return std::sqrt(sum / count);
    };

    const double llevel = windowed_mean(0, window_len);
    const double rlevel = windowed_mean(sigmoid_len - window_len, window_len);
    const double srange = std::abs(llevel - rlevel);

    bookends[0] = 0;
    bookends[1] = sigmoid_len - 1;
    if (srange == 0.0) {
        return;
    }

    const double lsdev = windowed_sdev(0, window_len, llevel);
    const double rsdev = windowed_sdev(sigmoid_len - window_len, window_len, rlevel);
    const double noise = std::max(lsdev, rsdev);
    const double thresh = thresh_sigma * noise;
    //const double thresh = thresh_frac * srange;

    for (int i = 1; i < sigmoid_len; ++i) {
        if (std::abs(sigmoid[i] - llevel) <= thresh) {
            bookends[0] = i;
        } else {
            break;
        }
    }

    for (int i = sigmoid_len - 2; i > bookends[0]; --i) {
        if (std::abs(sigmoid[i] - rlevel) <= thresh) {
            bookends[1] = i;
        } else {
            break;
        }
    }
}

/**
 * @brief Find the samples at the start and end of a sigmoid's linear segment
 *
 * Sigmoid should start high and end low.
 *
 * @param sigmoid Uniformly, ordered samples along a sigmoid
 * @param sigmoid_len Number of samples in `sigmoid`
 * @param prop_lo Ratio required to detect
 * @param prop_hi Ratio required to detect
 * @param res Output buffer. res[0] is the start bookend, res[1] is the end bookend
 */
template <typename T>
static void sigmoid_bookends(
    T *sigmoid,
    const int sigmoid_len,
    const double prop_descent,
    const double prop_lo,
    const double prop_hi,
    int res[2]
) {
    res[0] = (sigmoid_len == 0 ? 0 : -1);  // -1 := unset
    res[1] = (sigmoid_len == 0 ? 0 : -1);

    bool found_left = false;
    for (int s = 0; s < sigmoid_len; ++s) {
        found_left = (s > 0 && sigmoid[s] < prop_descent * sigmoid[s - 1]);

        // Select final bounds when either we're at the last sample or the
        // current sample is smaller than the three preceding samples
        bool thrice_decreased = (
            s > 3
            && sigmoid[s] >= sigmoid[s - 1]
            && sigmoid[s] >= sigmoid[s - 2]
            && sigmoid[s] >= sigmoid[s - 3]
        );
        if (s != (sigmoid_len - 1) && !(found_left && thrice_decreased)) {
            continue;
        }

        // Find highest and lowest sample in the sigmoid
        double min_val = sigmoid[0];
        double max_val = sigmoid[0];
        for (int j = 1; j < s; ++j) {
            min_val = (min_val < sigmoid[j] ? min_val : sigmoid[j]);
            max_val = (max_val > sigmoid[j] ? max_val : sigmoid[j]);
        }

        //
        const double thresh_lo = (max_val - min_val) * prop_lo + min_val;
        const double thresh_hi = (max_val - min_val) * prop_hi + min_val;
        for (int j = 0; j < s; ++j) {
            if (res[0] == -1 && sigmoid[j] < thresh_hi) {
                res[0] = j;
            }
            if (j != (s - 1) && !(res[0] != -1 && sigmoid[j] < thresh_lo)) {
                continue;
            }
            if (res[0] == -1) {  // use first value if haven't set min_val yet
                res[0] = 0;
            }
            res[1] = j;
            return;
        }
    }
}

template <typename Z>
int32_t run(AppArgs &args) {

    // Logger
    Logger log(true, args.logfile);
    logger::set_global(&log);

    log_args(args);

    // Sweep datafile output
    std::ofstream sweep_file;
    sweep_file.open(args.outdir + "/sweep.txt");  // TODO See if can recover thread ID


    //-------------------------------------------------------------------------
    // Read Problem
    //-------------------------------------------------------------------------

    qap::ReplicaPool annealer_h = qap::ReplicaPool<FloatingPoint, Z>::from_file(
        args.input_file, args.pool_width, args.pool_depth, 1, 1
    );
    annealer_h.shuffle_positions();
    qap::ReplicaPool annealer_d = annealer_h.cu_copy(cudaMemcpyHostToDevice);

    // Initialize
    annealer_d.recompute_force_mats(32);
    annealer_d.template recompute_energies<32>();

    annealer_d.cu_sync(annealer_h, cudaMemcpyDeviceToHost);  // TODO Use .on_device in method

    //-------------------------------------------------------------------------
    // Run Algorithm: Beta Sweep
    //-------------------------------------------------------------------------

    std::random_device dev;
    auto host_rng = std::mt19937(dev());

    Moment start_time = Clock::now();
    Moment check_time;

    // Beta Sweep
    int32_t best_score_sweep = INT32_MAX;
    FloatingPoint min_beta = 0;
    FloatingPoint max_beta = 0;
    if (args.do_beta_sweep) {  // TODO Might be able to optimize a bit (memory xfer)  // TODO Make its own function

        // Warmup
        const FloatingPoint autostep = exp2(
            (log2(kDefaultBetaMax) - kMinBetaToTry) / (float)(annealer_d.pool_width - 1)
        );
        //const FloatingPoint autostep = pow(
        //    kDefaultBetaMax / kMinBetaToTry, 1.0 / (float)(annealer_d.pool_width - 1)
        //);
        annealer_d.regenerate_beta(max_beta, autostep);
        run_annealing_loop<Z>(annealer_d, args, 1, true, start_time, sweep_file, INT32_MAX);
        annealer_d.cu_sync(annealer_h, cudaMemcpyDeviceToHost);

        // 1) Sweep through candidate betas and record scores for each
        int32_t sweep_min = INT32_MAX;
        FloatingPoint energy_sums[1024] = {0};
        for (int i = 0; i < kNumBetaStep; ++i) {
            const FloatingPoint candidate_beta = exp2(kMinBetaToTry + i * kBetaStep);

            int32_t replica_scores[annealer_h.num_replica] = {0};
            int32_t last_best_score = INT32_MAX;
            int32_t best_score[1] = {INT32_MAX};

            const uint64_t zero = 0;

            reset_best_scores<Z>(annealer_d);
            for (int trial = 0; trial < kNumBetaTrials; ++trial) {
                // Re-setup
                if (kRandomTrialInit) {
                    annealer_h.shuffle_positions();
                    annealer_h.cu_sync(annealer_d, cudaMemcpyHostToDevice);
                    annealer_d.recompute_force_mats(32);
                    annealer_d.template recompute_energies<32>();
                } else {
                    annealer_h.cu_sync(annealer_d, cudaMemcpyHostToDevice);  // "checkpoint"
                }

                // Need to add these to ensure independent seeding
                double z = std::generate_canonical<double, std::numeric_limits<double>::digits>(
                    host_rng
                );
                uint64_t trial_seed = z * UINT64_MAX;
                cudaMemcpy(
                    annealer_d.graph_seeds,
                    &trial_seed,
                    sizeof(uint64_t),
                    cudaMemcpyHostToDevice
                );
                cudaMemcpy(
                    annealer_d.graph_step,
                    &zero,
                    sizeof(uint64_t),
                    cudaMemcpyHostToDevice
                );

                annealer_d.regenerate_beta(candidate_beta, 1);  // all same beta

                // Run trial
                run_annealing_loop<Z>(annealer_d, args, 10, true, start_time, sweep_file, INT32_MAX);

                // Track score
                annealer_d.thunk_scores(
                    best_score,
                    nullptr,
                    nullptr,
                    replica_scores
                );
                for (int r = 0; r < annealer_d.num_replica; ++r) {
                    energy_sums[i] += replica_scores[r];
                }
                if (*best_score < last_best_score) {
                    last_best_score = *best_score;
                }
                if (*best_score < best_score_sweep) {
                    best_score_sweep = *best_score;
                }
            }
            energy_sums[i] /= annealer_h.num_replica * kNumBetaTrials;
            logger::write(std::format(
                "Beta: {}, Energy: {}\n", candidate_beta, energy_sums[i]
            ));
            if (energy_sums[i] < sweep_min) {
                sweep_min = energy_sums[i];
            }
        }

        uint32_t min_max_indices[2] = {0, 0};
        if (kUseFullPlateauCheck) {
            sigmoid_bookends_alt(
                energy_sums,
                kNumBetaStep,
                min_max_indices
            );


            //for (int i = 1; i < kNumBetaStep; ++i) {
            //    if (energy_sums[i] < 0.9 * energy_sums[i - 1]) {
            //        min_max_indices[0] = i;
            //        min_max_indices[1] = i;
            //        break;
            //    }
            //}
            //for (int i = min_max_indices[0] + 1; i < kNumBetaStep; ++i) {
            //    if (
            //        i > 3
            //        && energy_sums[i] >= energy_sums[i - 1]
            //        && energy_sums[i] >= energy_sums[i - 2]
            //        && energy_sums[i] >= energy_sums[i - 3]
            //    ) {
            //        min_max_indices[1] = i;
            //        break;
            //    }
            //}
        } else {
            bool min_set = false;
            bool has_started_dec = false;
            int32_t best_e = energy_sums[0];
            for (int step = 1; step < kNumBetaStep; ++step) {
                int32_t step_e = energy_sums[step];
                if (step_e < best_e) {
                    best_e = step_e;
                    min_max_indices[1] = step;
                }
                if (best_e < 0.9 * energy_sums[step - 1]) {
                    has_started_dec = true;
                }
                if (step == kNumBetaStep - 1) {
                    int32_t max_e = INT32_MIN;
                    int32_t min_e = INT32_MAX;
                    uint32_t min_e_idx = 0;
                    for (int j = 0; j < step; ++j) {
                        if (energy_sums[j] < min_e) {
                            min_e = energy_sums[j];
                            min_e_idx = j;
                        }
                        if (j == 0) {
                            max_e = energy_sums[j];
                        }
                    }
                    for (int j =0; j < step; ++j) {
                        if (
                            !min_set
                            && (
                                ((j == min_e_idx - 1) && (j >= 1))
                                || (energy_sums[j] < (max_e - min_e) * args.beta_prop_hi + min_e)
                            )
                        ) {
                            min_max_indices[0] = j;
                            min_set = true;
                        }
                        if (
                            j == step - 1
                            || (
                                min_set
                                && (
                                    j == min_e_idx
                                    || (energy_sums[j] < (max_e - min_e) * args.beta_prop_lo + min_e)
                                )
                           )
                        ) {
                            if (!min_set) {
                                min_max_indices[0] = 0;
                            }
                            min_max_indices[1] = j;
                            break;
                        }
                    }
                }
            }
        }

        // Apply left/right cautions
        int64_t cautious_min = max(
            (int64_t)min_max_indices[0] - kSweepCautionLeft,
            (int64_t)0
        );
        int64_t cautious_max = min(
            (int64_t)min_max_indices[1] + kSweepCautionRight, (int64_t)(kNumBetaStep - 1)
        );
        min_max_indices[0] = (uint32_t)cautious_min;
        min_max_indices[1] = (uint32_t)cautious_max;

        min_beta = exp2(kMinBetaToTry + min_max_indices[0] * kBetaStep);
        max_beta = exp2(kMinBetaToTry + min_max_indices[1] * kBetaStep);

        //int min_max_indices[2] = {0};
        //sigmoid_bookends(
        //    energy_sums,
        //    kNumBetaStep,
        //    0.9,
        //    args.beta_prop_lo,
        //    args.beta_prop_hi,
        //    min_max_indices
        //);
        //float prev_energy = energy_sums[0];
        //int plateau_start = kNumBetaStep - 1;
        //for (int i = min_max_indices[0] + 1; i < kNumBetaStep; ++i) {
        //    if (energy_sums[i] >= prev_energy * 0.9995f) {  // TODO Magic number
        //        plateau_start = i - 1;
        //        break;
        //    }
        //    prev_energy = energy_sums[i];
        //}
        //min_max_indices[1] = plateau_start;
        //min_beta = exp2(kMinBetaToTry + min_max_indices[0] * kBetaStep);
        //max_beta = exp2(kMinBetaToTry + min_max_indices[1] * kBetaStep);

        logger::write(std::format("Selecting minimum beta of {}\n", min_beta));
        logger::write(std::format("Selecting maximum beta of {}\n", max_beta));

        check_time = Clock::now();
        logger::write(std::format(
            "Selecting Beta Parameters took {} seconds.\n",
            ms_passed(start_time, check_time) / 1000.0
        ));
    } else{
        max_beta = args.beta_max;
        min_beta = args.beta_min;
    }


    annealer_h.cu_sync(annealer_d, cudaMemcpyHostToDevice);
    annealer_d.recompute_force_mats(32);
    annealer_d.template recompute_energies<32>();

    //-------------------------------------------------------------------------
    // Run Algorithm: Parallel Annealing
    //-------------------------------------------------------------------------

    logger::write(std::format("Beta range: [{}, {}]\n", min_beta, max_beta));
    logger::write(std::format("Initial best score: {}\n", best_score_sweep));
    const FloatingPoint autostep = pow(
        max_beta / min_beta, 1.0 / (float)(annealer_d.pool_width - 1)
    );
    annealer_d.regenerate_beta(max_beta, autostep);
    reset_best_scores(annealer_d);
    int32_t best_score = run_annealing_loop(
        annealer_d,
        args,
        args.mega_iter_lim,
        false,
        start_time,
        sweep_file,
        best_score_sweep
    );

    // Total runtime
    check_time = Clock::now();
    auto run_duration_ms = ms_passed(start_time, check_time);

    //-------------------------------------------------------------------------
    // Wrap Up
    //-------------------------------------------------------------------------

    // Logging
    std::cout << "\nHalting program.\n";
    logger::write(std::format("Single best energy found: {}\n", best_score));
    logger::write(std::format("Elapsed time: {} seconds.\n", run_duration_ms / 1000.0));

    // Memory management
    //cudaStreamSynchronize(annealer_d.graph_stream);
    //annealer_d.cu_free();
    //annealer_h.cu_free();
    return 0;
}

int main(const int argc, const char *const argv[]) {

    //-------------------------------------------------------------------------
    // Parse Arguments
    //-------------------------------------------------------------------------

    auto args = argparse::parse<AppArgs>(argc, argv);
    validate_args(args);
    set_derived_args(args);

    //-------------------------------------------------------------------------
    // Setup Environment
    //-------------------------------------------------------------------------

    // Output directory
    if (!std::filesystem::create_directory(args.outdir)) {
        throw std::runtime_error(
            "output directory already exists or could not be created"
        );
    }

    if (args.large_energies) {
        return run<int64_t>(args);
    } else {
        return run<int32_t>(args);
    }
}
