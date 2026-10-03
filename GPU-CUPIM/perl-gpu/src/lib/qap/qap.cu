#ifndef PERL_QAP
#define PERL_QAP

/* cuda */
#include <cassert>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

/* standard */
#include <math.h>
#include <random>
#include <stdexcept>
#include <stdint.h>
#include <string>

/* local */
#include "layout.cu"
#include "random.cuh"

namespace qap {

    //=================================================================================
    // ReplicaPool Structure
    //=================================================================================

    /** @brief Collection of replicas for parallel simulated annealing */
    template <typename F, typename Z>
    struct ReplicaPool {
    public:
        /**
         * @brief How many replicas "wide" the pool is
         *
         * Replicas are arranged in two dimensions during initialization:
         *
         *     +------+    +------+    +------+
         *     |      |    |      |    |      |
         *   +------+ |  +------+ |  +------+ |
         *   |      |-+  |      |-+  |      |-+
         *   |      |    |      |    |      |
         *   +------+    +------+    +------+
         *
         * In the above example, the replica pool has width 3 and depth 2. Replicas at
         * the same width index may be referred to as a "deck". By default, all replicas
         * in a deck have the same beta value and are initialized with the same position
         * for each facility. By default, beta values decrease along the width axis.
         */
        uint32_t pool_width;

        /**
         * @brief How many replicas "deep" the pool is
         *
         * See ReplicaPool::pool_width for more details.
         */
        uint32_t pool_depth;

        /** Total number of replicas in the pool */
        uint32_t num_replica;

        /**
         * @brief Values shared by all replicas
         *
         * Every replica has the same problem size, flow matrix, and distance matrix.
         */
        Layout invariants;

        /**
         * @brief Position currently associated with a facility
         *
         * Index: (replica    , facility    )
         * Shape: (num_replica, problem_size)
         */
        Position *position_of_facility;

        /**
         * @brief Facility currently associated with a position
         *
         * Index: (replica    , position    )
         * Shape: (num_replica, problem_size)
         */
        Facility *facility_at_position;

        /**
         * @brief Beta values associated with each replica
         *
         * Index: (replica    ,)
         * Shape: (num_replica,)
         */
        F *betas;
        F *betas_staging;  // always host-side

        /**
         * @brief Time since the indexed replicas has been swapped
         */
        int32_t *time_since_swap_beta;
        
        //--------------------------------------------------
        // Intermediate Buffers
        //--------------------------------------------------

        /**
         * @brief Random numbers uniformly sampled in the range [0, 1]
         *
         * Buffer is repopulated at the start of every annealing iteration (an iteration
         * is a processing of all positions/facilities once).
         *
         * Index: (replica    , position)
         * Shape: (num_replica, problem_size)
         *
         * Actual shape in memory is padded up to the nearest multiple of 4 due to how
         * number generation works
         */
        float *random_uniform;

        /**
         * @brief Force matrices for each replica
         *
         * Entry `force_mat[r][i][a]` stores the force exerted on position `i` by
         * facility `a`. These values are kept updated during annealing.
         *
         * Index: (replica    , position    , facility    )
         * Shape: (num_replica, problem_size, problem_size)
         */
        Z *force_mat;
        Z *force_mat_transposed;  // transposed matrices for coallesced loads

        /**
         * @brief Energy deltas of each potential swap of a given facility
         *
         * This buffer only has meaning in the context of
         * `ReplicaPool::calc_energy_deltas_and_swap_probs(Position pi)`, where
         * `energy_deltas[r][pj]` indicates the change in energy from swapping the
         * facilities in positions pi and pj with each other.
         *
         * Index: (replica    , position    )
         * Shape: (num_replica, problem_size)
         */
        int32_t *energy_deltas;

        /**
         * @brief Swap probabilities for each potential swap of a given facility
         *
         * Identical to ReplicaPool::energy_deltas, but stores the probability of each
         * swap instead of the energy change.
         *
         * Is also used to store the cumulative probability at certain stages of the
         * algorithm.
         *
         * Index: (replica    , position    )
         * Shape: (num_replica, problem_size)
         */
        F *swap_probs;

        /** @brief Swaps with resolved collisions */
        uint32_t *swap_targets;

        //--------------------------------------------------
        // One-Shot Stuff
        //--------------------------------------------------

        /**
         * @brief Requested swaps (with collisions possible)
         *
         * Index: (replica    , position    )
         * Shape: (num_replica, problem_size)
         */
        uint32_t *proposed_targets;

        /**
         * @brief Energy delta associated with each (replica, pi)'s proposal
         *
         * Index: (replica    , position    )
         * Shape: (num_replica, problem_size)
         */
        int32_t *proposed_deltas;
        //int32_t *partials_scratch;  // (num_replica, ps, ps)

        // Ballots at each position. High 32 bits are dE, low are claimant pi
        uint64_t *position_owner;

        //--------------------------------------------------
        // Score Buffers
        //--------------------------------------------------

        /**
         * @brief Current energy of each replica
         *
         * Index: (replica    ,)
         * Shape: (num_replica,)
         */
        int32_t *replica_score;

        /**
         * @brief Best energy that each replica has achieved
         *
         * Index: (replica    ,)
         * Shape: (num_replica,)
         */
        int32_t *best_score_per_replica;

        /**
         * @brief Best energy that each deck has achieved
         *
         * Index: (width     ,)
         * Shape: (pool_width,)
         */
        int32_t *best_score_per_deck;

        /**
         * @brief Best energy that the pool has achieved
         *
         * Index: ()
         * Shape: scalar
         */
        int32_t *best_score_overall;
        int32_t *best_score_host;  // for pinning

        // Manual throttling to stop runaway CPU
        cudaEvent_t score_event;
        static constexpr uint32_t kThrottleDepth = 8;
        cudaEvent_t manual_throttle[kThrottleDepth];
        uint64_t throttle_counter = 0;

        bool on_device;

        //--------------------------------------------------
        // Graph Stuff
        //--------------------------------------------------

        /** @brief RNG seeds for Graph(s) */
        uint64_t *graph_seeds;

        /** @brief CudaGraph execution happens over */
        cudaGraph_t cuda_graph;
        cudaGraphExec_t graph_exec;

        /** @brief Graph flag for "one-shot" variation */
        bool graph_ready;

        /** @brief Graph flag for serial variation, only used if GibbsPatience set */
        bool graph_ready_ss;
        cudaStream_t graph_stream;
        uint64_t *graph_step;

        //--------------------------------------------------
        // Methods
        //--------------------------------------------------

        // CUDA utilities
        inline void cu_sync(ReplicaPool<F, Z> &dst, const cudaMemcpyKind direction);
        inline ReplicaPool<F, Z> cu_copy(const cudaMemcpyKind direction);
        inline void cu_free();

        // Other utilities
        static inline ReplicaPool<F, Z> from_file(
            const std::string &filename,
            const uint32_t pool_width,
            const uint32_t pool_depth,
            const F beta0,
            const F beta_step
        );

        // Algorithm
        inline void shuffle_positions();
        inline void recompute_force_mats(const uint32_t tile_len);

        /**
         * @brief Fully recompute all replica energies
         *
         * This operation is expensive and should only be called during initialization
         * or in special circumstances. During annealing, use update_energies.
         */
        template <uint32_t tTileLen>
        inline void recompute_energies();

        /** @brief Serial pass over all vertices */
        template <uint32_t tNumThread>
        inline void perform_step(const uint64_t seed);

        /** @brief "One shot" pass over all vertices using a snapshot */
        template <uint32_t tNumThread>
        inline void perform_step_one_shot(const uint64_t seed);

        /** @brief Move updated scores over to host */
        inline void thunk_scores(
            int32_t *best_score,
            int32_t *all_scores,
            int32_t *best_replica_scores,
            int32_t *current_replica_scores
        );

        /** @brief Single step of bubble sort over beta value assignments */
        template <uint32_t tThreadRoot>
        inline void percolate_betas();

        /** @brief Manually assign new beta values to each replica (respecting decks) */
        inline void regenerate_beta(const long double beta0, const long double delta_beta);

        /** @brief Calculate scores for each deck */
        template <uint32_t tThreadNum>
        inline void recompute_deck_scores();
    };

    //=================================================================================
    // Setup Functions
    //=================================================================================

    //-------------------------------------------------------------------------
    // Shuffle Facility Positions
    //-------------------------------------------------------------------------

    /** @brief Randomly reassign facilities to positions */
    template <typename F, typename Z>
    void ReplicaPool<F, Z>::shuffle_positions() {
        if (this->on_device) {
            // TODO
            throw std::runtime_error("device-side shuffling not currently supported");
        }
        const uint32_t ps = this->invariants.problem_size;

        // Initialize RNG
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<> distr(0, ps - 1);

        // Attempt a swap for every position in each replica, keeping the same
        // structure at each depth across decks (i.e. each deck will have the
        // same positioning, with difference being in replicas). We do this for
        // a fair comparison of the impact of beta values.
        for (uint32_t d = 0; d < this->pool_depth; ++d) {
            for (uint32_t i = 0; i < ps; ++i) {

                // Randomly choose another position to swap with
                const uint32_t j = distr(rng);
                if (i == j) {
                    continue;
                }

                // Perform the same swap in each deck for the replica at depth d
                const uint32_t pw = this->pool_width;
                for (uint32_t w = 0; w < pw; ++w) {
                    const uint32_t r = (d * pw + w) * ps;
                    const uint32_t pi = r + i;
                    const uint32_t pj = r + j;
                    const uint32_t fa = r + this->facility_at_position[pi];
                    const uint32_t fb = r + this->facility_at_position[pj];
                    std::swap(
                        this->facility_at_position[pi], this->facility_at_position[pj]
                    );
                    std::swap(
                        this->position_of_facility[fa], this->position_of_facility[fb]
                    );
                }
            }
        }
    }

    //-------------------------------------------------------------------------
    // Initialize Force Matrix
    //-------------------------------------------------------------------------

    /**
     * @brief Kernel for `ReplicaPool::recompute_force_mats()`
     *
     * @tparam F Floating-point type
     * @param pool_d Replica pool to compute over
     */
    template <typename F, typename Z>
    __global__
    static void recompute_force_mats_d(ReplicaPool<F, Z> pool_d) {
        const uint32_t ps = pool_d.invariants.problem_size;

        //                  tile idx     tile len     tile cut-in
        const uint32_t pi = blockIdx.y * blockDim.y + threadIdx.y;  // force_mat row
        const uint32_t fa = blockIdx.x * blockDim.x + threadIdx.x;  // force_mat col

        bool in_bounds = (
            blockIdx.z < pool_d.num_replica  // blockIdx.z := replica_idx
            && pi < ps
            && fa < ps
        );
        if (!in_bounds) {
            return;
        }
        const uint32_t ia_global = (blockIdx.z * ps * ps) + (pi * ps) + fa;
        const uint32_t ia_global_transposed = (blockIdx.z * ps * ps) + (fa * ps) + pi;
        const uint32_t offset_1d = (blockIdx.z * ps);

        // NOTE Device-side sum reduction algorithms have too much overhead for these
        //      smaller summation loops (problem sizes smaller than warps). Minor gains
        //      could be found with unrolling
        pool_d.force_mat[ia_global] = 0;
        pool_d.force_mat_transposed[ia_global_transposed] = 0;
        Z force_delta = 0;
        for (int pj = 0; pj < pool_d.invariants.problem_size; ++pj) {

            // NOTE Hoisted: keep dist_xx loads in front of fb fetch
            const int32_t dist_ij = __ldg(&pool_d.invariants.dist_mat[pi * ps + pj]);
            const int32_t dist_ji = __ldg(&pool_d.invariants.dist_mat_transposed[pj + ps * pi]);

            const uint32_t fb = pool_d.facility_at_position[offset_1d + pj];

            const int32_t flow_aj = __ldg(&pool_d.invariants.flow_mat_transposed[fa + ps * fb]);
            const int32_t flow_bi = __ldg(&pool_d.invariants.flow_mat[fb * ps + fa]);

            force_delta -= ((Z)dist_ij * flow_aj + (Z)dist_ji * flow_bi);
        }
        pool_d.force_mat[ia_global] = force_delta;
        pool_d.force_mat_transposed[ia_global_transposed] = force_delta;
    }

    /**
     * @brief Compute force matrices from scratch
     *
     * Based on the current state of each replica, computes their force
     * matrices.
     *
     * This operation is very expensive and should only be run during
     * initialization or when you need to realign the force matrix due to drift
     * or other alterations to replicas not performed by the annealing pipeline
     * functions.
     *
     * @tparam F
     * @param pool_d
     * @return __global__
     */
    template <typename F, typename Z>
    void ReplicaPool<F, Z>::recompute_force_mats(const uint32_t tile_len) {
        if (this->on_device) {
            if (tile_len > 32) {
                throw std::runtime_error("tile length cannot exceed 32");
            }
            const uint32_t ps = this->invariants.problem_size;
            const uint32_t tiles_long = (ps + tile_len - 1) / tile_len;

            const dim3 grid(tiles_long, tiles_long, this->num_replica);
            const dim3 block(tile_len, tile_len);

            recompute_force_mats_d<<<grid, block>>>(*this);
        } else {
            // TODO: Integrate host-side operation in new API
            throw std::runtime_error("please use original code for host-side operation");
        }
    }

    //-------------------------------------------------------------------------
    // Initialize Energies
    //-------------------------------------------------------------------------

    /**
     * @brief Compute energy values associated with each i,j pair
     *
     * Intermediate step in computing the energy of a replica. Partial values
     * are summed together in `sum_energy_partials_thrust()`.
     *
     * @tparam F
     * @param pool_d
     * @param out_buf
     * @return __global__
     */
    template <typename F, typename Z>
    __global__
    static void compute_energy_partials(
        const ReplicaPool<F, Z> pool_d, int32_t *const out_buf
    ) {
        const uint32_t ps = pool_d.invariants.problem_size;

        //                  tile idx     tile len     tile cut-in
        const uint32_t pi = blockIdx.y * blockDim.y + threadIdx.y;  // row
        const uint32_t pj = blockIdx.x * blockDim.x + threadIdx.x;  // col

        bool in_bounds = (
            blockIdx.z < pool_d.num_replica  // blockIdx.z := replica_idx
            && pi < ps
            && pj < ps
        );
        if (!in_bounds) {
            return;
        }
        const uint32_t fa = pool_d.facility_at_position[blockIdx.z * ps + pi];
        const uint32_t fb = pool_d.facility_at_position[blockIdx.z * ps + pj];

        const int32_t dist_ij = __ldg(&pool_d.invariants.dist_mat[pi * ps + pj]);
        const int32_t flow_ab = __ldg(&pool_d.invariants.flow_mat[fa * ps + fb]);

        const uint32_t out_idx = (blockIdx.z * ps * ps) + (pi * ps) + pj;
        out_buf[out_idx] = dist_ij * flow_ab;
    }

    /**
     * @brief Calculate the energy of each replica.
     *
     * Must be run after `compute_energy_partials()` was run on the same
     * replica pool. Sums the partials values produced thereby, populating the
     * replica score buffer of the replica pool.
     *
     * @tparam F
     * @param pool_d
     * @param partials
     * @return __global__
     */
    template <typename F, typename Z>
    __global__
    static void sum_energy_partials_thrust(ReplicaPool<F, Z> pool_d, int32_t *partials) {
        const uint32_t ps = pool_d.invariants.problem_size;
        const int32_t replica_score = thrust::reduce(
            thrust::device,
            partials + ps * ps * blockIdx.x,
            partials + ps * ps * (blockIdx.x + 1)  // blockIdx.x := replica_idx
        );
        pool_d.replica_score[blockIdx.x] = replica_score;
    }

    /** Textbook tree reduction in shared memory */
    template <typename F, typename Z, uint32_t tBlockSize>
    __global__
    static void sum_energy_partials_d(ReplicaPool<F, Z> pool_d, int32_t *partials) {
        extern __shared__ int32_t sum_sh[];

        const uint32_t ps = pool_d.invariants.problem_size;

        const uint32_t repl_idx = blockIdx.x;
        const uint32_t thread_idx = threadIdx.x;

        const int32_t *base = partials + repl_idx * ps * ps;
        int32_t acc = 0;
        for (uint32_t i = thread_idx; i < ps * ps; i += tBlockSize) {
            acc += base[i];
        }
        sum_sh[thread_idx] = acc;
        __syncthreads();

        for (uint32_t s = tBlockSize / 2; s > 0; s >>= 1) {
            if (thread_idx < s) {
                sum_sh[thread_idx] += sum_sh[thread_idx + s];
            }
            if (s > 32) __syncthreads(); else __syncwarp();
        }
        if (thread_idx == 0) {
            pool_d.replica_score[repl_idx] = sum_sh[0];
        }
    }

    template <typename F, typename Z>
    __global__
    static void update_best_scores_d(ReplicaPool<F, Z> pool_d) {
        const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= pool_d.num_replica) {
            return;
        }
        const int32_t s = pool_d.replica_score[r];
        if (s < pool_d.best_score_per_replica[r]) {
            pool_d.best_score_per_replica[r] = s;
            atomicMin(pool_d.best_score_overall, s);
        }
    }

    /**
     * @brief Compute replica energies from scratch
     *
     * As with `ReplicaPool::recompute_force_mats()`, this is expensive and
     * should only be called during initialization or when alignment is
     * necessary due to drift or unaccounted modification of the replica pool.
     *
     * @tparam F Floating-point type
     * @tparam tTileLen Side length of tile for tiled CUDA operations
     */
    template <typename F, typename Z>
    template <uint32_t tTileLen>
    void ReplicaPool<F, Z>::recompute_energies() {
        if (tTileLen > 32) {
            throw std::runtime_error("tile length cannot exceed 32");
        }
        const uint32_t ps = this->invariants.problem_size;

        if (this->on_device) {
            cudaMemset(this->replica_score, 0, sizeof(int32_t) * this->num_replica);

            int32_t *partials;  // computed for every i, j pair
            cudaMalloc(&partials, sizeof(int32_t) * this->num_replica * ps * ps);

            // Compute energy partials for each (i, j)
            const uint32_t tiles_long = (ps + tTileLen - 1) / tTileLen;
            const dim3 grid(tiles_long, tiles_long, this->num_replica);
            const dim3 block(tTileLen, tTileLen);
            compute_energy_partials<<<grid, block>>>(*this, partials);
            sum_energy_partials_thrust<<<this->num_replica, 1>>>(*this, partials);
            cudaFree(partials);
        } else {
            // TODO: Integrate host-side operation in new API
            throw std::runtime_error("please use original code for host-side operation");
        }
        cudaDeviceSynchronize();
    }

    //=================================================================================
    // Annealing Algorithm
    //=================================================================================

    /**
     * @brief Calculate energy change from swapping the facilities at pi and pj
     *
     * @param pool Replica pool to operate over (must be on the device)
     * @param replica_idx Replica to operate over
     * @param pi Replica-relative position of facility to swap with the one in pj
     * @param pj Replica-relative position of facility to swap with the one in pi
     */
    template <typename F, typename Z>
    __device__ __host__
    int32_t calc_energy_delta(
        const ReplicaPool<F, Z> pool,
        const uint32_t replica_idx,
        const uint32_t pi,
        const uint32_t pj
    ) {
        const uint32_t ps = pool.invariants.problem_size;
        const uint32_t repl_offset_1d = replica_idx * ps;
        const uint32_t repl_offset_2d = replica_idx * ps * ps;

        // NOTE Hoisted: keep in front of fa/fb fetches
        const int32_t dist_ii = __ldg(&pool.invariants.dist_mat_diag[pi]);  // NxN
        const int32_t dist_jj = __ldg(&pool.invariants.dist_mat_diag[pj]);  // p p
        const int32_t dist_ij = __ldg(&pool.invariants.dist_mat[pi * ps + pj]);
        const int32_t dist_ji = __ldg(&pool.invariants.dist_mat_transposed[pj + ps * pi]);

        const uint32_t fa = __ldg(&pool.facility_at_position[repl_offset_1d + pi]);  // RxN
        const uint32_t fb = __ldg(&pool.facility_at_position[repl_offset_1d + pj]);

        const int32_t flow_aa = __ldg(&pool.invariants.flow_mat[fa * ps + fa]);  // NxN
        const int32_t flow_bb = __ldg(&pool.invariants.flow_mat[fb * ps + fb]);  // f f
        const int32_t flow_ab = __ldg(&pool.invariants.flow_mat[fa * ps + fb]);
        const int32_t flow_ba = __ldg(&pool.invariants.flow_mat[fb * ps + fa]);

        // `force_ia` corresponds to $I_{i}^{a}$ in the paper
        const Z force_ib = __ldg(&pool.force_mat[repl_offset_2d + pi * ps + fb]);  // RxNxN
        const Z force_ia = __ldg(&pool.force_mat[repl_offset_2d + pi * ps + fa]);  //   p f
        const Z force_ja = __ldg(&pool.force_mat_transposed[repl_offset_2d + pj + ps * fa]);
        const Z force_jb = __ldg(&pool.force_mat_transposed[repl_offset_2d + pj + ps * fb]);

        const Z force_term = force_ib + force_ja - force_ia - force_jb;
        const int32_t dist_term = dist_ii + dist_jj - dist_ij - dist_ji;
        const int32_t flow_term = flow_aa + flow_bb - flow_ab - flow_ba;

        return (force_term - dist_term * flow_term);
    }

    /**
     * @brief Compute dE and swap probabilities for each candidate swap
     *
     * Should be invoked with grid dimensions (num_blocks, num_replica), where
     * num_blocks is ceil(ps / num_threads).
     *
     * Modifies the following buffers:
     * - `pool_d.energy_deltas`
     * - `pool_d.swap_probs`
     */
    template <typename F, typename Z>
    __global__
    static void calc_energy_deltas_and_swap_probs_d(
        ReplicaPool<F, Z> pool_d, const uint32_t pi
    ) {
        const uint32_t ps = pool_d.invariants.problem_size;
        const uint32_t pj = blockIdx.x * blockDim.x + threadIdx.x;  // swap candidate
        const uint32_t global_idx = blockIdx.y * ps + pj;  // blockIdx.y := replica_idx

        if (pj >= ps) {
            return;
        }

        // NOTE Hoisted: keep this before `calc_energy_delta()` call for ILP
        const F beta = __ldg(&pool_d.betas[blockIdx.y]);
        const int32_t energy_delta = calc_energy_delta(pool_d, blockIdx.y, pi, pj);

        pool_d.energy_deltas[blockIdx.y * ps + pj] = energy_delta;
        pool_d.swap_probs[global_idx] = exp2(beta * energy_delta);
    }

    /**
     * @brief Primary phase of the swap operation
     *
     * Performs all the math for finding swap targets and updating force and
     * energy values.
     *
     * @tparam F Floating-point type
     * @tparam tProblemSizeFactor Largest denominator of N among [1,2,4]
     * @param pool_d Replica pool to operate over
     * @param pi Position currently being swapped
     */
    template <typename F, typename Z, uint32_t tProblemSizeFactor>
    __global__
    void perform_swaps_i_d(ReplicaPool<F, Z> pool_d, const uint32_t pi) {
        extern __shared__ int32_t flow_mat_sh[];

        const uint32_t ps = pool_d.invariants.problem_size;
        const uint32_t r1 = blockIdx.z * ps;
        const uint32_t r2 = blockIdx.z * ps * ps;  // blockIdx.z := replica_idx

        //                  tile idx     tile len     tile cut-in
        const uint32_t pk = blockIdx.y * blockDim.y + threadIdx.y;  // force_mat row
        const uint32_t fc = blockIdx.x * blockDim.x + threadIdx.x;  // force_mat col

        const uint32_t pj = __ldg(&pool_d.swap_targets[blockIdx.z]);
        const uint32_t fa = __ldg(&pool_d.facility_at_position[r1 + pi]);
        const uint32_t fb = __ldg(&pool_d.facility_at_position[r1 + pj]);  // broadcasts

        // Stage flow matrix into shared memory
        const uint32_t start = threadIdx.y * blockDim.x + threadIdx.x;
        const uint32_t stride = blockDim.x * blockDim.y;

        // 4 elements per read
        if (tProblemSizeFactor == 4) {
            const int4 *read_sh = reinterpret_cast<const int4*>(pool_d.invariants.flow_mat);
            int4 *write_sh = reinterpret_cast<int4*>(flow_mat_sh);
            for (uint32_t k = start; k < ps / 4; k += stride) {
                write_sh[k] = __ldg(&read_sh[fa * (ps / 4) + k]);
                write_sh[k + ps / 4] = __ldg(&read_sh[fb * (ps / 4) + k]);
            }
            // TODO Might be able to use `int3` as well, but not worth dealing with alignment
        }

        // 2 elements per read
        else if (tProblemSizeFactor == 2) {
            const int2 *read_sh = reinterpret_cast<const int2*>(pool_d.invariants.flow_mat);
            int2 *write_sh = reinterpret_cast<int2*>(flow_mat_sh);
            for (uint32_t k = start; k < ps / 2; k += stride) {
                write_sh[k] = __ldg(&read_sh[fa * (ps / 2) + k]);
                write_sh[k + ps / 2] = __ldg(&read_sh[fb * (ps / 2) + k]);
            }
        }

        // 1 element per read
        else {
            for (uint32_t k = start; k < ps; k += stride) {
                flow_mat_sh[k] = __ldg(&pool_d.invariants.flow_mat[fa * ps + k]);
                flow_mat_sh[k + ps] = __ldg(&pool_d.invariants.flow_mat[fb * ps + k]);
            }
        }
        __syncthreads();

        if (pk >= ps || fc >= ps) {
            return;
        }

        const int32_t dist_kj = __ldg(&pool_d.invariants.dist_mat[pk * ps + pj]);
        const int32_t dist_ki = __ldg(&pool_d.invariants.dist_mat[pk * ps + pi]);
        const int32_t dist_jk = __ldg(&pool_d.invariants.dist_mat[pj * ps + pk]);
        const int32_t dist_ik = __ldg(&pool_d.invariants.dist_mat[pi * ps + pk]);

        const int32_t flow_cb = __ldg(&pool_d.invariants.flow_mat_transposed[fc + ps * fb]);
        const int32_t flow_ca = __ldg(&pool_d.invariants.flow_mat_transposed[fc + ps * fa]);
        const int32_t flow_ac = flow_mat_sh[fc];
        const int32_t flow_bc = flow_mat_sh[fc + ps];

        Z x = (
            (Z)(dist_kj - dist_ki) * (flow_cb - flow_ca)
            + (Z)(dist_jk - dist_ik) * (flow_bc - flow_ac)
        );
        pool_d.force_mat[r2 + pk * ps + fc] += x;
        pool_d.force_mat_transposed[r2 + pk + ps * fc] += x;

        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            const int32_t this_score = (
                pool_d.replica_score[blockIdx.z] - pool_d.energy_deltas[r1 + pj]
            );
            pool_d.replica_score[blockIdx.z] = this_score;
            if (this_score < pool_d.best_score_per_replica[blockIdx.z]) {
                pool_d.best_score_per_replica[blockIdx.z] = this_score;
                atomicMin(pool_d.best_score_overall, this_score);
            }
        }
    }

    /**
     * @brief Bookkeeping step of swapping
     *
     * Must be called after `perform_swaps_i_d`
     *
     * @tparam F Floating-point type
     * @param pool_d Replica pool to operate over
     * @param pi Position being swapped
     */
    template <typename F, typename Z>
    __global__
    void perform_swaps_ii_d(ReplicaPool<F, Z> pool_d, const uint32_t pi) {
        const uint32_t repl_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (repl_idx >= pool_d.num_replica) {  // repl_idx := replica_idx
            return;
        }
        const uint32_t ps = pool_d.invariants.problem_size;
        const uint32_t r1 = ps * repl_idx;

        const uint32_t pj = pool_d.swap_targets[repl_idx];
        const uint32_t fa = pool_d.facility_at_position[r1 + pi];
        const uint32_t fb = pool_d.facility_at_position[r1 + pj];

        pool_d.facility_at_position[r1 + pi] = fb;
        pool_d.facility_at_position[r1 + pj] = fa;
        pool_d.position_of_facility[r1 + fa] = pj;
        pool_d.position_of_facility[r1 + fb] = pi;
    }

    /**
     * @brief Fused kernel for `ReplicaPool::perform_step()`
     *
     * @tparam F Floating-point type
     * @tparam tBlockSize CUDA block size to use
     * @param pool_d Replica pool to operate over
     * @param pi Position being swapped
     */
    template <typename F, typename Z, uint32_t tBlockSize>
    __global__
    static void perform_step_d(ReplicaPool<F, Z> pool_d, const uint32_t pi) {
        static_assert(
            (tBlockSize & (tBlockSize - 1)) == 0, "block size must be a power of two"
        );
        extern __shared__ unsigned char sh_raw[];
        F *sh = reinterpret_cast<F*>(sh_raw);  // |tBlockSize|
        const uint32_t ps = pool_d.invariants.problem_size;

        // dE and P[swap] kernel
        const uint32_t replica_idx = blockIdx.x;
        const uint32_t r1 = replica_idx * ps;
        const uint32_t thread_idx = threadIdx.x;
        const F beta = __ldg(&pool_d.betas[replica_idx]);

        int32_t energy_delta = 0;
        if (thread_idx < ps) {
            energy_delta = calc_energy_delta(
                pool_d, replica_idx, pi, thread_idx
            );
            pool_d.energy_deltas[r1 + thread_idx] = energy_delta;
        }

        int32_t *idx_sh = reinterpret_cast<int32_t*>(sh_raw);
        idx_sh[thread_idx] = (thread_idx < ps) ? energy_delta : INT32_MIN;
        __syncthreads();
        for (uint32_t s = tBlockSize / 2; s > 0; s >>= 1) {
            if (thread_idx < s) {
                int32_t other = idx_sh[thread_idx + s];
                if (other > idx_sh[thread_idx]) {
                    idx_sh[thread_idx] = other;
                }
            }
            if (s > 32) __syncthreads(); else __syncwarp();
        }
        __syncthreads();
        const int32_t max_delta = idx_sh[0];

        if (thread_idx < ps) {
            sh[thread_idx] = exp2(beta * static_cast<F>(energy_delta - max_delta));
        } else {
            sh[thread_idx] = F(0);
        }
        __syncthreads();

        // Cumulative sum kernel
        if (ps > 32) {
            for (uint32_t stride = 1; stride < tBlockSize; stride *= 2) {
                F val = (thread_idx >= stride) ? sh[thread_idx - stride] : F(0);
                __syncthreads();
                sh[thread_idx] += val;
                __syncthreads();
            }
        } else { // warp scan
            F val = (thread_idx < ps) ? sh[thread_idx] : F(0);
            for (uint32_t offset = 1; offset < 32; offset <<= 1) {
                F tmp = __shfl_up_sync(0xffffffff, val, offset);
                if (thread_idx >= offset) {
                    val += tmp;
                }
            }
            if (thread_idx < ps) {
                sh[thread_idx] = val;
            }
            __syncwarp();
        }

        // Threshold kernel
        if (thread_idx < ps) {
            const F rand_thresh = pool_d.random_uniform[r1 + pi] * sh[ps - 1];
            const bool found = (
                sh[thread_idx] >= rand_thresh
                && (thread_idx == 0 || sh[thread_idx - 1] < rand_thresh)
            );
            const bool fallback = (
                (thread_idx == ps - 1) && (sh[thread_idx] < rand_thresh)
            );
            if (found || fallback) {
                pool_d.swap_targets[replica_idx] = thread_idx;
            }
        }
    }

    static constexpr uint32_t adaptive_tile_len(const uint32_t ps) {
        // Removed v2.0 adaptivity because program was SM bound and
        // 16 was optimal tile len for given hardware. Might need to
        // tune for other GPUs - would need to see specs or profiling
        // data
        return 16;
    }

    /**
     * @brief Perform a single annealing step
     *
     * An annealing step consists of a pass over every position exactly once
     * each. At each position, a swap will be attempted and the replicas'
     * physical values updated.
     *
     * @tparam F Floating-point type
     * @tparam tNumThread Number of CUDA threads to use per block
     * @param seed Seed for RNG
     */
    template <typename F, typename Z>
    template <uint32_t tNumThread>
    void ReplicaPool<F, Z>::perform_step(const uint64_t seed) {
        const uint32_t ps = this->invariants.problem_size;
        static const uint32_t tile_len = adaptive_tile_len(ps);

        if (!this->graph_ready_ss) {
            cudaFuncSetAttribute(
                perform_step_d<F, Z, 32>,  // L1 bound
                cudaFuncAttributePreferredSharedMemoryCarveout,
                cudaSharedmemCarveoutMaxL1
            );  // below three are same as above
            cudaFuncSetAttribute(perform_step_d<F, Z, 64>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
            cudaFuncSetAttribute(perform_step_d<F, Z, 128>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
            cudaFuncSetAttribute(perform_step_d<F, Z, 256>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);

            cudaFuncSetAttribute(
                perform_swaps_i_d<F, Z, 1>,  // __shared__ bound
                cudaFuncAttributePreferredSharedMemoryCarveout,
                cudaSharedmemCarveoutMaxShared
            );  // below two are same as above
            cudaFuncSetAttribute(perform_swaps_i_d<F, Z, 2>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
            cudaFuncSetAttribute(perform_swaps_i_d<F, Z, 4>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);

            uint64_t init_seed = seed;
            cudaMemcpy(
                this->graph_seeds, &init_seed, sizeof(uint64_t), cudaMemcpyHostToDevice
            );

            // Start capture
            cudaStreamBeginCapture(this->graph_stream, cudaStreamCaptureModeRelaxed);

            // `perform_step()`
            const uint32_t nb = (
                (this->num_replica * ps + 4 * tNumThread - 1) / (4 * tNumThread)
            );
            generate_random_numbers<<<nb, tNumThread, 0, this->graph_stream>>>(
                this->num_replica * ps,
                this->graph_seeds,
                this->graph_step,
                (float4*)this->random_uniform
            );
            for (uint32_t pi = 0; pi < ps; ++pi) {
                // Inline everything to avoid device capture issues

                if (ps <= 32) {
                    perform_step_d<F, Z, 32><<<
                        this->num_replica, 32, 32 * sizeof(F), this->graph_stream
                    >>>(*this, pi);
                } else if (ps <= 64) {
                    perform_step_d<F, Z, 64><<<
                        this->num_replica, 64, 64 * sizeof(F), this->graph_stream
                    >>>(*this, pi);
                } else if (ps <= 128) {
                    perform_step_d<F, Z, 128><<<
                        this->num_replica, 128, 128 * sizeof(F), this->graph_stream
                    >>>(*this, pi);
                } else if (ps <= 256) {
                    perform_step_d<F, Z, 256><<<
                        this->num_replica, 256, 256 * sizeof(F), this->graph_stream
                    >>>(*this, pi);
                } else {
                    // TODO Handle N > 256 with tiling
                    throw std::runtime_error("N cannot exceed 256");
                }

                const uint32_t tiles_long = (ps + tile_len - 1) / tile_len;
                const dim3 grid_i(tiles_long, tiles_long, this->num_replica);
                const dim3 block_i(tile_len, tile_len);

                // Heavy-lifing swap step
                const size_t smem_size = 2 * ps * sizeof(int32_t);
                if (ps % 4 == 0) {
                    perform_swaps_i_d<F, Z, 4><<<
                        grid_i, block_i, smem_size, this->graph_stream
                    >>>(*this, pi);
                }
                else if (ps % 2 == 0) {
                    perform_swaps_i_d<F, Z, 2><<<
                        grid_i, block_i, smem_size, this->graph_stream
                    >>>(*this, pi);
                }
                else {
                    perform_swaps_i_d<F, Z, 1><<<
                        grid_i, block_i, smem_size, this->graph_stream
                    >>>(*this, pi);
                }

                // Bookkeeping swap step
                const uint32_t block_size_ii = 32;
                const uint32_t grid_size_ii = (
                    (this->num_replica + block_size_ii - 1) / block_size_ii
                );
                perform_swaps_ii_d<<<
                    grid_size_ii, block_size_ii, 0, this->graph_stream
                >>>(*this, pi);
            }

            // Advance RNG as graph-tracked operation
            const uint64_t s = (uint64_t)nb * tNumThread;
            advance_rng<<<1, 1, 0, this->graph_stream>>>(this->graph_step, s);

            // End capture
            cudaStreamEndCapture(this->graph_stream, &this->cuda_graph);
            cudaGraphInstantiate(
                &this->graph_exec, this->cuda_graph, nullptr, nullptr, 0
            );
            this->graph_ready_ss = true;
        }

        // Manully throttle to ensure GPU has runway
        if (this->throttle_counter >= kThrottleDepth) {
            cudaEvent_t &old_slot = this->manual_throttle[this->throttle_counter % kThrottleDepth];
            cudaEventSynchronize(old_slot);
        }
        cudaGraphLaunch(this->graph_exec, this->graph_stream);

        // Bookkeeping manual throttle
        cudaEvent_t &new_slot = this->manual_throttle[this->throttle_counter % kThrottleDepth];
        cudaEventRecord(new_slot, this->graph_stream);
        ++this->throttle_counter;
    }

    /** Propose swaps with collisions */
    template <typename F, typename Z, uint32_t tThreadLoad>
    __global__
    static void os_pwc_d(ReplicaPool<F, Z> pool_d) {
        const uint32_t ps = pool_d.invariants.problem_size;
        const uint32_t pi = blockIdx.x;
        const uint32_t repl_idx = blockIdx.y;
        const uint32_t thread_idx = threadIdx.x;
        const uint32_t r1 = repl_idx * ps;

        const F beta = __ldg(&pool_d.betas[repl_idx]);

        // Was the dE and swap probs kernel in v1.0
        int loc_deltas[tThreadLoad];
        #pragma unroll
        for (uint32_t k = 0; k < tThreadLoad; ++k) {
            const uint32_t pj = thread_idx + k * 32;
            if (pj < ps) {
                loc_deltas[k] = calc_energy_delta(pool_d, repl_idx, pi, pj);
            } else {
                loc_deltas[k] = 0;
            }
        }

        int32_t local_max = INT32_MIN;
        #pragma unroll
        for (uint32_t k = 0; k < tThreadLoad; ++k) {
            const uint32_t pj = thread_idx + k * 32;
            if (pj < ps && loc_deltas[k] > local_max) {
                local_max = loc_deltas[k];
            }
        }
        #pragma unroll
        for (uint32_t d = 1; d < 32; d <<= 1) {
            int32_t v = __shfl_xor_sync(0xffffffff, local_max, d);
            if (v > local_max) {
                local_max = v;
            }
        }
        const int32_t global_max_delta = local_max;

        F loc_swap_probs[tThreadLoad];
        #pragma unroll
        for (uint32_t k = 0; k < tThreadLoad; ++k) {
            const uint32_t pj = thread_idx + k * 32;
            if (pj < ps) {
                loc_swap_probs[k] = exp2(
                    beta * static_cast<F>(loc_deltas[k] - global_max_delta)
                );
            } else {
                loc_swap_probs[k] = F(0);
            }
        }

        // Was the summation kernel in v1.0
        F local_total = loc_swap_probs[0];
        #pragma unroll
        for (uint32_t k = 1; k < tThreadLoad; ++k) {
            loc_swap_probs[k] += local_total;
            local_total = loc_swap_probs[k];
        }
        F warp_offset;
        {
            F x = local_total;
            #pragma unroll
            for (uint32_t d = 1; d < 32; d <<= 1) {
                F v = __shfl_up_sync(0xffffffff, x, d);
                if (thread_idx >= d) {
                    x += v;
                }
            }
            warp_offset = x - local_total;
        }
        const F grand_total = __shfl_sync(0xffffffff, warp_offset + local_total, 31);

        // Was the threshold kernel in v1.0
        const F rand_thresh = pool_d.random_uniform[r1 + pi] * grand_total;
        uint32_t winner_pj = 0;
        int32_t winner_delta = loc_deltas[0];
        bool found = false;

        #pragma unroll
        for (uint32_t k = 0; k < tThreadLoad; ++k) {
            const uint32_t pj = thread_idx + k * 32;
            if (pj >= ps) {
                break;
            }
            const F cur  = warp_offset + loc_swap_probs[k];
            const F prev = (k == 0) ? warp_offset : warp_offset + loc_swap_probs[k - 1];
            if (cur >= rand_thresh && prev < rand_thresh) {
                winner_pj  = pj;
                winner_delta = loc_deltas[k];
                found = true;
            }
        }
        const uint32_t found_mask = __ballot_sync(0xffffffff, found);
        if (found_mask != 0) {
            const uint32_t first_winner = __ffs(found_mask) - 1;
            if (thread_idx == first_winner) {
                pool_d.proposed_targets[r1 + pi] = winner_pj;
                pool_d.proposed_deltas [r1 + pi] = winner_delta;
            }
        } else {
            const uint32_t fallback_idx = (ps - 1) % 32;
            const uint32_t fallback_k = (ps - 1) / 32;
            if (thread_idx == fallback_idx) {
                pool_d.proposed_targets[r1 + pi] = ps - 1;
                pool_d.proposed_deltas [r1 + pi] = loc_deltas[fallback_k];
            }
        }
    }

    /** Initialize ballot race */
    template <typename F, typename Z>
    __global__
    static void os_resolve_i_d(ReplicaPool<F, Z> pool_d) {
        const uint32_t ps = pool_d.invariants.problem_size;
        const uint32_t replica_idx = blockIdx.x;
        const uint32_t pi = blockIdx.y * blockDim.x + threadIdx.x;
        if (pi >= ps) {
            return;
        }
        pool_d.position_owner[replica_idx * ps + pi] = UINT64_MAX;
    }

    /** Ballot race */
    template <typename F, typename Z>
    __global__
    static void os_resolve_ii_d(ReplicaPool<F, Z> pool_d) {
        const uint32_t ps = pool_d.invariants.problem_size;
        const uint32_t replica_idx = blockIdx.x;
        const uint32_t pi = blockIdx.y * blockDim.x + threadIdx.x;

        if (pi >= ps) {
            return;
        }
        const uint32_t r1 = replica_idx * ps;
        const uint32_t pj = pool_d.proposed_targets[r1 + pi];

        if (pi == pj) {
            return;
        }
        const int32_t raw_delta = pool_d.proposed_deltas[r1 + pi];

        // Energy-favoring ballot resolution
        //
        // Things I tried that didn't work (see commit 25c5bbb if curious):
        // - Clamped ballots to reduce starvation
        // - Coin toss (converged about 20% slower)
        // - Check sign only (never converged)
        const uint32_t biased = ~(static_cast<uint32_t>(raw_delta) ^ 0x80000000u);
        const uint64_t ballot = (
            (static_cast<uint64_t>(biased) << 32) | static_cast<uint64_t>(pi)
        );
        atomicMin(  // target
            reinterpret_cast<unsigned long long*>(&pool_d.position_owner[r1 + pj]),
            ballot
        );
        atomicMin(  // source
            reinterpret_cast<unsigned long long*>(&pool_d.position_owner[r1 + pi]),
            ballot
        );
    }

    /** Ballot race resolution */
    template <typename F, typename Z>
    __global__
    static void os_resolve_iii_d(ReplicaPool<F, Z> pool_d) {
        const uint32_t ps = pool_d.invariants.problem_size;
        const uint32_t replica_idx = blockIdx.x;
        const uint32_t pi = blockIdx.y * blockDim.x + threadIdx.x;

        if (pi >= ps) {
            return;
        }
        const uint32_t r1 = replica_idx * ps;
        const uint32_t pj = pool_d.proposed_targets[r1 + pi];

        if (pi == pj) {
            return;
        }
        const uint32_t winner_pi = static_cast<uint32_t>(pool_d.position_owner[r1 + pi] & 0xFFFFFFFFu);
        const uint32_t winner_pj = static_cast<uint32_t>(pool_d.position_owner[r1 + pj] & 0xFFFFFFFFu);
        if (winner_pi != pi || winner_pj != pi) {
            pool_d.proposed_targets[r1 + pi] = ps;  // mark rejected
        }
    }

    template <typename F, typename Z, uint32_t tProblemSizeFactor>
    __global__
    static void os_apply_swaps_i_d(ReplicaPool<F, Z> pool_d) {
        extern __shared__ int32_t flow_mat_sh[];

        const uint32_t ps = pool_d.invariants.problem_size;

        const uint32_t replica_idx = blockIdx.z / ps;
        const uint32_t pi = blockIdx.z % ps;

        const uint32_t r1 = replica_idx * ps;
        const uint32_t r2 = replica_idx * ps * ps;

        const uint32_t pj = pool_d.proposed_targets[r1 + pi];
        if (pj >= ps) {
            return;  // rejected is marked with ps
        }

        //                  tile idx     tile len     tile cut-in
        const uint32_t pk = blockIdx.y * blockDim.y + threadIdx.y;  // force_mat row
        const uint32_t fc = blockIdx.x * blockDim.x + threadIdx.x;  // force_mat col

        const uint32_t fa = __ldg(&pool_d.facility_at_position[r1 + pi]);
        const uint32_t fb = __ldg(&pool_d.facility_at_position[r1 + pj]);

        const uint32_t start = threadIdx.y * blockDim.x + threadIdx.x;
        const uint32_t stride = blockDim.x * blockDim.y;

        // Vectorized loads
        if (tProblemSizeFactor == 4) {
            const int4 *read_sh = reinterpret_cast<const int4*>(pool_d.invariants.flow_mat);
            int4 *write_sh = reinterpret_cast<int4*>(flow_mat_sh);
            for (uint32_t k = start; k < ps / 4; k += stride) {
                write_sh[k] = __ldg(&read_sh[fa * (ps / 4) + k]);
                write_sh[k + ps / 4] = __ldg(&read_sh[fb * (ps / 4) + k]);
            }
        } // below same as above
        else if (tProblemSizeFactor == 2) {
            const int2 *read_sh = reinterpret_cast<const int2*>(pool_d.invariants.flow_mat);
            int2 *write_sh = reinterpret_cast<int2*>(flow_mat_sh);
            for (uint32_t k = start; k < ps / 2; k += stride) {
                write_sh[k] = __ldg(&read_sh[fa * (ps / 2) + k]);
                write_sh[k + ps / 2] = __ldg(&read_sh[fb * (ps / 2) + k]);
            }
        } else {
            for (uint32_t k = start; k < ps; k += stride) {
                flow_mat_sh[k] = __ldg(&pool_d.invariants.flow_mat[fa * ps + k]);
                flow_mat_sh[k + ps] = __ldg(&pool_d.invariants.flow_mat[fb * ps + k]);
            }
        }
        __syncthreads();

        if (pk >= ps || fc >= ps) {
            return;
        }
        const int32_t dist_kj = __ldg(&pool_d.invariants.dist_mat[pk * ps + pj]);
        const int32_t dist_ki = __ldg(&pool_d.invariants.dist_mat[pk * ps + pi]);
        const int32_t dist_jk = __ldg(&pool_d.invariants.dist_mat[pj * ps + pk]);  // TODO Consider transposition
        const int32_t dist_ik = __ldg(&pool_d.invariants.dist_mat[pi * ps + pk]);

        const int32_t flow_cb = __ldg(&pool_d.invariants.flow_mat_transposed[fc + ps * fb]);
        const int32_t flow_ca = __ldg(&pool_d.invariants.flow_mat_transposed[fc + ps * fa]);
        const int32_t flow_ac = flow_mat_sh[fc];
        const int32_t flow_bc = flow_mat_sh[fc + ps];

        const Z x = (
            (dist_kj - dist_ki) * (flow_cb - flow_ca)
            + (dist_jk - dist_ik) * (flow_bc - flow_ac)
        );

        atomicAdd(&pool_d.force_mat[r2 + pk * ps + fc], x);
        atomicAdd(&pool_d.force_mat_transposed[r2 + pk + ps * fc], x);

        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            const int32_t delta = pool_d.proposed_deltas[r1 + pi];
            const int32_t old_score = atomicAdd(&pool_d.replica_score[replica_idx], -delta);
            const int32_t new_score = old_score - delta;
            if (new_score < pool_d.best_score_per_replica[replica_idx]) {
                pool_d.best_score_per_replica[replica_idx] = new_score;
                atomicMin(pool_d.best_score_overall, new_score);
            }
        }
    }

    template <typename F, typename Z>
    __global__
    static void os_apply_swaps_ii_d(ReplicaPool<F, Z> pool_d) {
        const uint32_t ps = pool_d.invariants.problem_size;

        const uint32_t replica_idx = blockIdx.x;
        const uint32_t pi = blockIdx.y * blockDim.x + threadIdx.x;

        if (pi >= ps) {
            return;
        }
        const uint32_t r1 = replica_idx * ps;
        const uint32_t pj = pool_d.proposed_targets[r1 + pi];

        if (pj >= ps || pi == pj) {
            return;
        }
        const uint32_t fa = pool_d.facility_at_position[r1 + pi];
        const uint32_t fb = pool_d.facility_at_position[r1 + pj];

        pool_d.facility_at_position[r1 + pi] = fb;
        pool_d.facility_at_position[r1 + pj] = fa;
        pool_d.position_of_facility[r1 + fa] = pj;
        pool_d.position_of_facility[r1 + fb] = pi;
    }

    template <typename F, typename Z, uint32_t tPSFactor>
    __global__
    static void os_update_force_mats_d(ReplicaPool<F, Z> pool_d) {
        extern __shared__ int32_t flow_sh[]; // (ps * ps,)

        const uint32_t ps = pool_d.invariants.problem_size;
        const uint32_t replica_idx = blockIdx.z;
        const uint32_t r1 = replica_idx * ps;
        const uint32_t r2 = replica_idx * ps * ps;

        const uint32_t pk = blockIdx.y * blockDim.y + threadIdx.y;  // force_mat row
        const uint32_t fc = blockIdx.x * blockDim.x + threadIdx.x;  // force_mat col

        const uint32_t start = threadIdx.y * blockDim.x + threadIdx.x;
        const uint32_t stride = blockDim.x * blockDim.y;

        // Accumulate in registers
        Z acc = 0;
        for (uint32_t pi = 0; pi < ps; ++pi) {
            const uint32_t pj = pool_d.proposed_targets[r1 + pi];
            if (pj >= ps) {
                continue;
            }
            const uint32_t fa = __ldg(&pool_d.facility_at_position[r1 + pi]);
            const uint32_t fb = __ldg(&pool_d.facility_at_position[r1 + pj]);

            // Vectorized loads into shared
            if (tPSFactor == 4) {
                const int4 *fsrc = reinterpret_cast<const int4*>(
                    pool_d.invariants.flow_mat
                );
                int4 *fdst = reinterpret_cast<int4*>(flow_sh);
                for (uint32_t k = start; k < ps / 4; k += stride) {
                    fdst[k] = __ldg(&fsrc[fa * (ps / 4) + k]);
                    fdst[k + ps / 4] = __ldg(&fsrc[fb * (ps / 4) + k]);
                }
            } // below same as above
            else if (tPSFactor == 2) {
                const int2 *fsrc = reinterpret_cast<const int2*>(pool_d.invariants.flow_mat);
                int2 *fdst = reinterpret_cast<int2*>(flow_sh);
                for (uint32_t k = start; k < ps / 2; k += stride) {
                    fdst[k] = __ldg(&fsrc[fa * (ps / 2) + k]);
                    fdst[k + ps / 2] = __ldg(&fsrc[fb * (ps / 2) + k]);
                }
            } else {
                for (uint32_t k = start; k < ps; k += stride) {
                    flow_sh[k] = __ldg(&pool_d.invariants.flow_mat[fa * ps + k]);
                    flow_sh[k + ps] = __ldg(&pool_d.invariants.flow_mat[fb * ps + k]);
                }
            }
            __syncthreads();

            if (pk < ps && fc < ps) {
                const int32_t dist_kj = __ldg(&pool_d.invariants.dist_mat[pk * ps + pj]);
                const int32_t dist_ki = __ldg(&pool_d.invariants.dist_mat[pk * ps + pi]);
                const int32_t dist_jk = __ldg(&pool_d.invariants.dist_mat[pj * ps + pk]);  // TODO Consider transposition
                const int32_t dist_ik = __ldg(&pool_d.invariants.dist_mat[pi * ps + pk]);

                const int32_t flow_ca = __ldg(&pool_d.invariants.flow_mat_transposed[fc + ps * fa]);
                const int32_t flow_cb = __ldg(&pool_d.invariants.flow_mat_transposed[fc + ps * fb]);
                const int32_t flow_ac = flow_sh[fc];
                const int32_t flow_bc = flow_sh[fc + ps];

                acc += (
                    (dist_kj - dist_ki) * (flow_cb - flow_ca)
                    + (dist_jk - dist_ik) * (flow_bc - flow_ac)
                );
            }
            __syncthreads();
        }

        // Blockwise so doesn't need to be atomic (see formula in notes)
        if (pk < ps && fc < ps) {
            pool_d.force_mat[r2 + pk * ps + fc] += acc;
            pool_d.force_mat_transposed[r2 + pk + ps * fc] += acc;
        }
    }

    /**
     * @brief Compute exact score after swaps based on snapshot
     *
     * Targets only swapped rows/columns. See notes for math.
     */
    template <typename F, typename Z, uint32_t tBlockSize>
    __global__
    static void os_score_from_force_mat_d(ReplicaPool<F, Z> pool_d) {
        // TODO Add templated static if guard
        //extern __shared__ int32_t scores_sh[];  // (tBlockSize,)
        extern __shared__ uint8_t scores_raw_sh[];
        Z *scores_sh = reinterpret_cast<Z*>(scores_raw_sh);

        const uint32_t ps = pool_d.invariants.problem_size;
        const uint32_t repl_idx = blockIdx.x;
        const uint32_t thread_idx = threadIdx.x;
        const uint32_t r1 = repl_idx * ps;
        const uint32_t r2 = repl_idx * ps * ps;

        // Gather op w/ zero-padding for out of range
        Z x = 0;
        if (thread_idx < ps) {
            const uint32_t fa = pool_d.facility_at_position[r1 + thread_idx];
            x = pool_d.force_mat[r2 + thread_idx * ps + fa];
        }
        scores_sh[thread_idx] = x;
        __syncthreads();

        // Tree reduce
        for (uint32_t s = tBlockSize / 2; s > 0; s >>= 1) {
            if (thread_idx < s) {
                scores_sh[thread_idx] += scores_sh[thread_idx + s];
            }
            if (s > 32) __syncthreads(); else __syncwarp();
        }

        // Correct scores
        if (thread_idx == 0) {
            const int32_t new_score = -scores_sh[0] / 2;
            pool_d.replica_score[repl_idx] = new_score;
            if (new_score < pool_d.best_score_per_replica[repl_idx]) {
                pool_d.best_score_per_replica[repl_idx] = new_score;
                atomicMin(pool_d.best_score_overall, new_score);
            }
        }
    }

    /** Uses snapshot and collision resolution instead of iterative updates */
    template <typename F, typename Z>
    template <uint32_t tNumThread>
    void ReplicaPool<F, Z>::perform_step_one_shot(const uint64_t seed) {
        const uint32_t ps = this->invariants.problem_size;
        static const uint32_t tile_len = adaptive_tile_len(ps);

        if (!this->graph_ready) {
            cudaFuncSetAttribute(
                os_pwc_d<F, Z, 1>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                cudaSharedmemCarveoutMaxL1
            ); // below all same as above
            cudaFuncSetAttribute(os_pwc_d<F, Z, 2>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
            cudaFuncSetAttribute(os_pwc_d<F, Z, 3>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
            cudaFuncSetAttribute(os_pwc_d<F, Z, 4>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
            cudaFuncSetAttribute(os_pwc_d<F, Z, 5>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
            cudaFuncSetAttribute(os_pwc_d<F, Z, 6>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
            cudaFuncSetAttribute(os_pwc_d<F, Z, 7>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
            cudaFuncSetAttribute(os_pwc_d<F, Z, 8>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);

            uint64_t init_seed = seed;
            cudaMemcpy(
                this->graph_seeds, &init_seed, sizeof(uint64_t), cudaMemcpyHostToDevice
            );
            cudaStreamBeginCapture(this->graph_stream, cudaStreamCaptureModeRelaxed);

            // RNG
            const uint32_t nb =
                (this->num_replica * ps + 4 * tNumThread - 1) / (4 * tNumThread)
            ;
            generate_random_numbers<<<nb, tNumThread, 0, this->graph_stream>>>(
                this->num_replica * ps,
                this->graph_seeds,
                this->graph_step,
                (float4*)this->random_uniform
            );

            // Proposals with collisions
            {
                const dim3 pwc_grid(ps, this->num_replica);
                const uint32_t thread_load = (ps + 31) / 32;
                if (thread_load == 1) {
                    os_pwc_d<F, Z, 1><<<pwc_grid, 32, 0, this->graph_stream>>>(*this);
                } else if (thread_load == 2) {
                    os_pwc_d<F, Z, 2><<<pwc_grid, 32, 0, this->graph_stream>>>(*this);
                } else if (thread_load == 3) {
                    os_pwc_d<F, Z, 3><<<pwc_grid, 32, 0, this->graph_stream>>>(*this);
                } else if (thread_load == 4) {
                    os_pwc_d<F, Z, 4><<<pwc_grid, 32, 0, this->graph_stream>>>(*this);
                } else if (thread_load == 5) {
                    os_pwc_d<F, Z, 5><<<pwc_grid, 32, 0, this->graph_stream>>>(*this);
                } else if (thread_load == 6) {
                    os_pwc_d<F, Z, 6><<<pwc_grid, 32, 0, this->graph_stream>>>(*this);
                } else if (thread_load == 7) {
                    os_pwc_d<F, Z, 7><<<pwc_grid, 32, 0, this->graph_stream>>>(*this);
                } else if (thread_load == 8) {
                    os_pwc_d<F, Z, 8><<<pwc_grid, 32, 0, this->graph_stream>>>(*this);
                } else {
                    throw std::runtime_error("N must be LEQ 256");
                }
            } // NOTE scoping hints to compiler for hopeful register gains

            // Resolve collisions
            {
                const uint32_t num_tiles = (ps + 255) / 256;
                const dim3 rc_grid(this->num_replica, num_tiles);
                os_resolve_i_d<<<rc_grid, 256, 0, this->graph_stream>>>(*this);
                os_resolve_ii_d<<<rc_grid, 256, 0, this->graph_stream>>>(*this);
                os_resolve_iii_d<<<rc_grid, 256, 0, this->graph_stream>>>(*this);
            }

            // Update force matrix etc
            {
                const uint32_t num_tiles = (ps + tile_len - 1) / tile_len;
                const dim3 fm_grid(num_tiles, num_tiles, this->num_replica);
                const dim3 fm_block(tile_len, tile_len);
                const size_t smem_sz = 2 * ps * sizeof(int32_t);
                if (ps % 4 == 0) {
                    os_update_force_mats_d<F, Z, 4><<<
                        fm_grid, fm_block, smem_sz, this->graph_stream
                    >>>(*this);
                } else if (ps % 2 == 0) {
                    os_update_force_mats_d<F, Z, 2><<<
                        fm_grid, fm_block, smem_sz, this->graph_stream
                    >>>(*this);
                } else {
                    os_update_force_mats_d<F, Z, 1><<<
                        fm_grid, fm_block, smem_sz, this->graph_stream
                    >>>(*this);
                }
            }

            // Finalize swaps and bookkeep
            {
                const uint32_t num_tiles = (ps + 256 - 1) / 256;
                const dim3 swap_grid(this->num_replica, num_tiles);
                os_apply_swaps_ii_d<<<swap_grid, 256, 0, this->graph_stream>>>(*this);
            }

            // Update score
            {
                uint32_t score_block = 32;
                while (score_block < ps) {
                    score_block <<= 1;
                }
                const uint32_t smem_sz = score_block * sizeof(Z);
                if (score_block <= 32) {
                    os_score_from_force_mat_d<F, Z, 32><<<
                        this->num_replica, 32, smem_sz, this->graph_stream
                    >>>(*this);
                } // below are same as above
                else if (score_block <= 64) {
                    os_score_from_force_mat_d<F, Z, 64><<<this->num_replica, 64, smem_sz, this->graph_stream>>>(*this);
                } else if (score_block <= 128) {
                    os_score_from_force_mat_d<F, Z, 128><<<this->num_replica, 128, smem_sz, this->graph_stream>>>(*this);
                } else if (score_block <= 256) {
                    os_score_from_force_mat_d<F, Z, 256><<<this->num_replica, 256, smem_sz, this->graph_stream>>>(*this);
                } else if (score_block <= 512) {
                    os_score_from_force_mat_d<F, Z, 512><<<this->num_replica, 512, smem_sz, this->graph_stream>>>(*this);
                } else {
                    os_score_from_force_mat_d<F, Z, 1024><<<this->num_replica, 1024, smem_sz, this->graph_stream>>>(*this);
                }
            }

            // Advance RNG
            const uint64_t rng_stride = (uint64_t)nb * tNumThread;
            advance_rng<<<1, 1, 0, this->graph_stream>>>(this->graph_step, rng_stride);

            cudaStreamEndCapture(this->graph_stream, &this->cuda_graph);
            cudaGraphInstantiate(
                &this->graph_exec, this->cuda_graph, nullptr, nullptr, 0
            );
            this->graph_ready = true;
        }

        // Throttle: ensure the GPU has runway
        if (this->throttle_counter >= kThrottleDepth) {
            cudaEvent_t &old_slot = (
                this->manual_throttle[this->throttle_counter % kThrottleDepth]
            );
            cudaEventSynchronize(old_slot);
        }
        cudaGraphLaunch(this->graph_exec, this->graph_stream);
        cudaEvent_t &new_slot = (
            this->manual_throttle[this->throttle_counter % kThrottleDepth]
        );
        cudaEventRecord(new_slot, this->graph_stream);
        ++this->throttle_counter;
    }

    //=================================================================================
    // Thunk Scores
    //=================================================================================

    /**
     * @brief Kernel for `ReplicaPool<F>::recompute_deck_scores()`
     *
     * @tparam F Floating-point type
     * @param pool_d Replica pool to compute over
     */
    template <typename F, typename Z>
    __global__
    static void recompute_deck_scores_d(ReplicaPool<F, Z> pool_d) {
        const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
        if (w >= pool_d.pool_width) {
            return;
        }
        int32_t best_score = INT32_MAX;
        for (int d = 0; d < pool_d.pool_depth; ++d) {  // TODO confirm axis
            const uint32_t replica_idx = d * pool_d.pool_width + w;
            const int32_t candidate_score = pool_d.replica_score[replica_idx];
            if (candidate_score < best_score) {
                best_score = candidate_score;
            }
        }
        pool_d.best_score_per_deck[w] = best_score;
    }

    /**
     * @brief Compute the best scores per deck in the replica pool
     *
     * @tparam F Floating-point type
     * @tparam tNumThreads Number of threads per CUDA block to use
     */
    template <typename F, typename Z>
    template <uint32_t tNumThreads>
    void ReplicaPool<F, Z>::recompute_deck_scores() {
        if (tNumThreads < this->pool_width) {
            throw std::runtime_error(
                "thunking deck scores not currently supported for width > thread count"
            );
        }

        // NOTE Not very efficient, but unwilling to move anything off the GPU here
        recompute_deck_scores_d<<<1, tNumThreads>>>(*this);
    }

    /**
     * @brief Place replica scores into buffers
     *
     * If any buffer is `nullptr`, the associated score will not be calculated.
     *
     * @tparam F Floating-point type
     * @param best_score Buffer for the best score found overall so far
     * @param best_deck_scores Buffer for the best deck score found so far
     * @param best_replica_scores Buffer for best per-replica scores so far
     * @param current_replica_scores Buffer for current replica scores
     */
    template <typename F, typename Z>
    void ReplicaPool<F, Z>::thunk_scores(
        int32_t *best_score,
        int32_t *best_deck_scores,
        int32_t *best_replica_scores,
        int32_t *current_replica_scores
    ) {
        cudaEventRecord(this->score_event, this->graph_stream);
        cudaEventSynchronize(this->score_event);
        //const bool score_posted = (cudaEventQuery(this->score_event) == cudaSuccess);

        const uint32_t tThreadRoot = 32;  // TODO Make param
        if (tThreadRoot > 32) {
            throw std::runtime_error("thread root cannot exceed 32");
        }
        if (!this->on_device) {
            // TODO
            throw std::runtime_error("thunking scores not currently supported for host!");
        }

        if (best_score != nullptr) {
         //   if (score_posted) {
                *best_score = *this->best_score_host;
          //  }
        }
        if (best_deck_scores != nullptr) {
            this->recompute_deck_scores<1024>();  // TODO axis might be wrong
            cudaMemcpy(
                best_deck_scores,
                this->best_score_per_deck,
                sizeof(int32_t) * this->pool_width,
                this->on_device ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost
            );
        }
        if (best_replica_scores != nullptr) {
            cudaMemcpy(
                best_replica_scores,
                this->best_score_per_replica,
                sizeof(int32_t) * this->num_replica,
                this->on_device ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost
            );
        }
        if (current_replica_scores != nullptr) {
            cudaMemcpy(
                current_replica_scores,
                this->replica_score,
                sizeof(int32_t) * this->num_replica,
                this->on_device ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost
            );
        }
    }

    //=================================================================================
    // Update Betas
    //=================================================================================

    /**
     * @brief Regenerate the replica pool's beta values
     *
     * Assigns betas such that all replicas at depth i have beta value of
     * `beta0 / (beta_delta ^ i)`.
     *
     * @tparam F Floating-point type
     * @param beta0 Highest beta assigned
     * @param beta_delta Divisor between adjacent beta values
     */
    template <typename F, typename Z>
    void ReplicaPool<F, Z>::regenerate_beta(const long double beta0, const long double beta_delta) {
        for (int d = 0; d < this->pool_depth; ++d) {
            F beta = beta0;
            for (int w = this->pool_width - 1; w >= 0; --w) {
                this->betas_staging[d * this->pool_width + w] = beta;  // TODO Move to gpu?
                beta /= beta_delta;
            }
        }
        cudaMemcpy(
            this->betas,
            this->betas_staging,
            this->num_replica * sizeof(F),
            this->on_device ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost
        );
    }

    /**
     * @brief Kernel called by `ReplicaPool<F>::percolate_betas()`
     *
     * @tparam F Floating-point type
     * @param pool_d Replica pool to update
     */
    template <typename F, typename Z>
    __global__
    void percolate_betas_d(ReplicaPool<F, Z> pool_d) {
        // TODO There's probably a better parallel algorithm for this

        const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
        if (d >= pool_d.pool_depth) {
            return;
        }


        pool_d.time_since_swap_beta[d]++;

        // NOTE We consciously decided not to do metropolis here (wasn't making
        //      a meaningful difference in performance)
        for (int w = 0; w < pool_d.pool_width - 1; ++w) {
            //float noise = pool_d.random_uniform[d * pool_d.pool_width + w];

            uint32_t idx1 = d * pool_d.pool_width + w;
            uint32_t idx2 = d * pool_d.pool_width + w + 1;

            int32_t e1 = pool_d.replica_score[idx1];
            int32_t e2 = pool_d.replica_score[idx2];

            //bool randswap = (
            //    noise < exp2((pool_d.betas[idx2] - pool_d.betas[idx1]) * (e2 - e1))
            //);
            //if (e1 < e2 || randswap) {  // NOTE Found not to make a difference
            if (e1 < e2) {
                F tmp = pool_d.betas[idx1];
                pool_d.betas[idx1] = pool_d.betas[idx2];
                pool_d.betas[idx2] = tmp;
                if( w ==pool_d.pool_width-1){
                    pool_d.time_since_swap_beta[d]=0;
                }
            }
        }

        if(pool_d.time_since_swap_beta[d] > 100){
            uint32_t idx1 = d * pool_d.pool_width + (pool_d.pool_width - 1);
            uint32_t idx2 = d * pool_d.pool_width + 0;

            F tmp = pool_d.betas[idx1];
            pool_d.betas[idx1] = pool_d.betas[idx2];
            pool_d.betas[idx2] = tmp;
            pool_d.time_since_swap_beta[d] = 0;
        }
    }

    /**
     * @brief Reassign betas
     *
     * Performs a single bubble sort step (not a complete sort) which assigns
     * lower temperatures to lower-score replicas.
     *
     * @tparam F Floating-point type
     * @tparam tThreadNum number of cuda threads per block
     */
    template <typename F, typename Z>
    template <uint32_t tThreadNum>
    void ReplicaPool<F, Z>::percolate_betas() {
        const uint32_t nb = (this->pool_depth + tThreadNum - 1) / tThreadNum;
        percolate_betas_d<<<nb, tThreadNum, 0, this->graph_stream>>>(*this);
    }

}  /* namespace qap */

#include "replica_pool_utils.cu"

#endif  /* PERL_QAP */
