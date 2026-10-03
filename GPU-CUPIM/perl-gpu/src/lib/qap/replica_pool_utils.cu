#include "qap.cu"

template <typename F, typename Z>
using ReplicaPool = qap::ReplicaPool<F, Z>;

//======================================================================================
//
//======================================================================================

/**
 * @brief Instantiate a replica pool from a problem file
 *
 * @param pool_width Number of replicas in a descending beta streak
 * @param pool_depth Number of replicas with the same beta
 */
template <typename F, typename Z>
ReplicaPool<F, Z> ReplicaPool<F, Z>::from_file(
    const std::string &filename,
    const uint32_t pool_width,
    const uint32_t pool_depth,
    const F beta0,
    const F beta_step
) {
    const uint32_t num_replica = pool_width * pool_depth;
    ReplicaPool<F, Z> new_pool = {
        .pool_width = pool_width,
        .pool_depth = pool_depth,
        .num_replica = num_replica,
        .invariants = Layout::from_file(filename),
        .on_device = false,
        .graph_ready = false,
        .graph_ready_ss = false,
    };
    const uint32_t ps = new_pool.invariants.problem_size;

    new_pool.time_since_swap_beta = new int32_t[1]; // {} should initialized to zero
    new_pool.time_since_swap_beta[0] = 0;

    // graph stuff
    new_pool.graph_step = new uint64_t[1];
    new_pool.graph_step[0] = 0;
    new_pool.graph_seeds = new uint64_t[1];
    new_pool.graph_seeds[0] = 0;
    cudaStreamCreate(&new_pool.graph_stream);

    new_pool.position_of_facility = new Position[ps * num_replica];
    new_pool.facility_at_position = new Facility[ps * num_replica];
    new_pool.betas = new F[num_replica];
    new_pool.betas_staging = new F[num_replica];

    // Intermediate buffers
    new_pool.random_uniform = new float[4 * ((num_replica * ps + 3) / 4)];
    new_pool.force_mat = new Z[num_replica * ps * ps];
    new_pool.force_mat_transposed = new Z[num_replica * ps * ps];
    new_pool.energy_deltas = new int32_t[num_replica * ps];
    new_pool.swap_probs = new F[num_replica * ps];
    new_pool.swap_targets = new uint32_t[num_replica];

    // One-shot stuff
    new_pool.proposed_targets = new uint32_t[num_replica * ps];
    new_pool.proposed_deltas = new int32_t[num_replica * ps];
    new_pool.position_owner = new uint64_t[num_replica * ps];
    //new_pool.partials_scratch = new int32_t[num_replica * ps * ps];

    // Score buffers
    new_pool.replica_score = new int32_t[num_replica];
    new_pool.best_score_per_replica = new int32_t[num_replica];
    new_pool.best_score_per_deck = new int32_t[pool_width];
    new_pool.best_score_overall = new int32_t[1];

    // Initialize betas with decrementing values
    for (int d = 0; d < pool_depth; ++d) {
        F beta = beta0;
        for (int w = pool_width - 1; w >= 0; --w) {
            new_pool.betas[d * pool_width + w] = beta;
            beta /= beta_step;
        }
    }

    // Initialize positions and facilities in ascending order
    for (int r = 0; r < num_replica; ++r) {
        for (int i = 0; i < ps; ++i) {
            new_pool.position_of_facility[r * ps + i] = i;
            new_pool.facility_at_position[r * ps + i] = i;
        }
    }

    // Initialize best scores to the worst score we can represent
    for (int w = 0; w < pool_width; ++w) {
        new_pool.best_score_per_deck[w] = INT32_MAX;
        for (int d = 0; d < pool_depth; ++d) {
            const uint32_t r = d * pool_width + w;
            new_pool.best_score_per_replica[r] = INT32_MAX;
            new_pool.replica_score[r] = 0;
        }
    }
    new_pool.best_score_overall[0] = INT32_MAX;

    // TODO Less hacky way of handling double-free in cu_free
    delete[] new_pool.invariants.facility_at_position;
    delete[] new_pool.invariants.position_of_facility;
    new_pool.invariants.facility_at_position = new uint32_t[0];
    new_pool.invariants.position_of_facility = new uint32_t[0];

    return new_pool;
}

//======================================================================================
// Utilities, CUDA
//======================================================================================

/** Sync the contents of this replica pool into a destination replica pool */
template <typename F, typename Z>
void ReplicaPool<F, Z>::cu_sync(ReplicaPool<F, Z> &dst, const cudaMemcpyKind direction) {
    const uint32_t ps = this->invariants.problem_size;

    if (dst.num_replica != this->num_replica) {
        throw std::runtime_error("cannot sync with differing replica counts");
    }
    dst.pool_depth = this->pool_depth;
    dst.pool_width = this->pool_width;
    this->invariants.cu_sync(dst.invariants, direction);
    
    cudaMemcpy(
        dst.time_since_swap_beta,
        this->time_since_swap_beta,
        sizeof(int32_t) * 1,
        direction
    );
    
    cudaMemcpy(
        dst.position_of_facility,
        this->position_of_facility,
        sizeof(uint32_t) * this->num_replica * ps,
        direction
    );
    cudaMemcpy(
        dst.facility_at_position,
        this->facility_at_position,
        sizeof(uint32_t) * this->num_replica * ps,
        direction
    );
    cudaMemcpy(dst.betas, this->betas, sizeof(F) * this->num_replica, direction);

    // Intermediate buffers
    cudaMemcpy(
        dst.random_uniform,
        this->random_uniform,
        sizeof(float) * ((this->num_replica * ps + 3) / 4) * 4,
        direction
    );
    cudaMemcpy(
        dst.force_mat,
        this->force_mat,
        sizeof(Z) * this->num_replica * ps * ps,
        direction
    );
    cudaMemcpy(
        dst.force_mat_transposed,
        this->force_mat_transposed,
        sizeof(Z) * this->num_replica * ps * ps,
        direction
    );
    cudaMemcpy(
        dst.energy_deltas,
        this->energy_deltas,
        sizeof(int32_t) * this->num_replica * ps,
        direction
    );
    cudaMemcpy(
        dst.swap_targets,
        this->swap_targets,
        sizeof(uint32_t) * this->num_replica,
        direction
    );
    cudaMemcpy(
        dst.swap_probs,
        this->swap_probs,
        sizeof(F) * this->num_replica * ps,
        direction
    );

    // One-shot stuff
    cudaMemcpy(
        dst.proposed_targets,
        this->proposed_targets,
        sizeof(uint32_t) * this->num_replica * ps,
        direction
    );
    cudaMemcpy(
        dst.proposed_deltas,
        this->proposed_deltas,
        sizeof(int32_t) * this->num_replica * ps,
        direction
    );
    cudaMemcpy(
        dst.position_owner,
        this->position_owner,
        sizeof(uint64_t) * this->num_replica * ps,
        direction
    );
    //cudaMemcpy(dst.partials_scratch, this->partials_scratch, sizeof(int32_t) * this->num_replica * ps * ps, direction);

    // Score buffers
    cudaMemcpy(
        dst.replica_score,
        this->replica_score,
        this->num_replica * sizeof(int32_t),
        direction
    );
    cudaMemcpy(
        dst.best_score_per_replica,
        this->best_score_per_replica,
        this->num_replica * sizeof(int32_t),
        direction
    );
    cudaMemcpy(
        dst.best_score_per_deck,
        this->best_score_per_deck,
        this->pool_width * sizeof(int32_t),
        direction
    );
    cudaMemcpy(
        dst.best_score_overall,
        this->best_score_overall,
        sizeof(int32_t),
        direction
    );
    cudaMemcpy(dst.graph_step, this->graph_step, sizeof(uint64_t), direction);
}

/** Create a copy of the replica pool. Will need to be match with a call to cu_free */
template <typename F, typename Z>
ReplicaPool<F, Z> ReplicaPool<F, Z>::cu_copy(const cudaMemcpyKind direction) {
    ReplicaPool<F, Z> new_pool = {
        .pool_width = this->pool_width,
        .pool_depth = this->pool_depth,
        .num_replica = this->num_replica,
        .invariants = this->invariants.cu_copy(direction),
        .graph_ready = false,
        .graph_ready_ss = false,
    };
    const uint32_t ps = this->invariants.problem_size;
    const uint32_t size_1d = ps * this->num_replica;
    const uint32_t size_2d = ps * size_1d;

    new_pool.betas_staging = new F[this->num_replica];  // always host-side

    switch (direction) {
        case cudaMemcpyHostToHost:
        case cudaMemcpyDeviceToHost:
            new_pool.time_since_swap_beta = new int32_t[1];
            new_pool.time_since_swap_beta[0] = 0;

            new_pool.on_device = false;
            new_pool.position_of_facility = new Position[size_1d];
            new_pool.facility_at_position = new Facility[size_1d];
            new_pool.betas = new F[this->num_replica];

            // Graph stuff
            new_pool.graph_seeds = new uint64_t[1];
            new_pool.graph_step = new uint64_t[1];
            new_pool.graph_seeds[0] = 0;
            new_pool.graph_stream = nullptr;

            // Intermediate buffers
            new_pool.random_uniform = new float[4 * ((size_1d + 3) / 4)];
            new_pool.force_mat = new Z[size_2d];
            new_pool.force_mat_transposed = new Z[size_2d];
            new_pool.energy_deltas = new int32_t[size_1d];
            new_pool.swap_targets = new uint32_t[this->num_replica];
            new_pool.swap_probs = new F[size_1d];

            // One-shot stuff
            new_pool.proposed_targets = new uint32_t[size_1d];
            new_pool.proposed_deltas = new int32_t[size_1d];
            new_pool.position_owner = new uint64_t[size_1d];
            //new_pool.partials_scratch = new int32_t[size_2d];

            // Score buffers
            new_pool.replica_score = new int32_t[this->num_replica];
            new_pool.best_score_per_replica = new int32_t[this->num_replica];
            new_pool.best_score_per_deck = new int32_t[this->pool_width];
            new_pool.best_score_overall = new int32_t[1];
            break;
        case cudaMemcpyHostToDevice:
        case cudaMemcpyDeviceToDevice:
            new_pool.on_device = true;


            cudaMalloc(
                &new_pool.time_since_swap_beta,
                sizeof(int32_t) * 1
            );

            cudaMalloc(
                &new_pool.position_of_facility,
                sizeof(uint32_t) * this->num_replica * ps
            );
            cudaMalloc(
                &new_pool.facility_at_position,
                sizeof(uint32_t) * this->num_replica * ps
            );
            cudaMalloc(&new_pool.betas, sizeof(F) * this->num_replica);

            // Graph stuff
            cudaMalloc(&new_pool.graph_seeds, sizeof(uint64_t));
            cudaMalloc(&new_pool.graph_step, sizeof(uint64_t));
            cudaMemset(new_pool.graph_seeds, 0, sizeof(uint64_t));
            cudaMemset(new_pool.graph_step, 0, sizeof(uint64_t));
            cudaStreamCreate(&new_pool.graph_stream);

            // Intermediate buffers
            cudaMalloc(
                &new_pool.random_uniform,
                sizeof(float) * 4 * ((this->num_replica * ps + 3) / 4)
            );
            cudaMalloc(
                &new_pool.force_mat,
                sizeof(Z) * this->num_replica * ps * ps
            );
            cudaMalloc(
                &new_pool.force_mat_transposed,
                sizeof(Z) * this->num_replica * ps * ps
            );
            cudaMalloc(
                &new_pool.energy_deltas,
                sizeof(int32_t) * this->num_replica * ps
            );
            cudaMalloc(
                &new_pool.swap_targets,
                sizeof(uint32_t) * this->num_replica
            );
            cudaMalloc(
                &new_pool.swap_probs,
                sizeof(F) * this->num_replica * ps
            );

            // One-shot stuff
            cudaMalloc(
                &new_pool.proposed_targets,
                sizeof(uint32_t) * this->num_replica * ps
            );
            cudaMalloc(
                &new_pool.proposed_deltas,
                sizeof(int32_t) * this->num_replica * ps
            );
            cudaMalloc(
                &new_pool.position_owner,
                sizeof(uint64_t) * this->num_replica * ps
            );
            //cudaMalloc(&new_pool.partials_scratch, sizeof(int32_t) * this->num_replica * ps * ps);

            // Score buffers
            cudaMalloc(
                &new_pool.replica_score,
                sizeof(int32_t) * this->num_replica
            );
            cudaMalloc(
                &new_pool.best_score_per_replica,
                sizeof(int32_t) * this->num_replica
            );
            cudaMalloc(
                &new_pool.best_score_per_deck,
                sizeof(int32_t) * this->pool_width
            );
            cudaHostAlloc(
                &new_pool.best_score_host,
                sizeof(int32_t),
                cudaHostAllocMapped
            );
            *new_pool.best_score_host = INT32_MAX;
            cudaHostGetDevicePointer(
                &new_pool.best_score_overall, new_pool.best_score_host, 0
            );
            cudaEventCreateWithFlags(&new_pool.score_event, cudaEventDisableTiming);
            for (int i = 0; i < kThrottleDepth; ++i) {
                cudaEventCreateWithFlags(
                    &new_pool.manual_throttle[i], cudaEventDisableTiming
                );
            }
            new_pool.throttle_counter = 0;
            break;
        default:
            throw std::runtime_error("unrecognized cudaMemcpyKind");
    }
    this->cu_sync(new_pool, direction);
    return new_pool;
}

/** Free replica pool from memory */
template <typename F, typename Z>
void ReplicaPool<F, Z>::cu_free() {
    this->invariants.cu_free();
    delete[] this->betas_staging;  // always host-side
    if (on_device) {
        cudaFree(this->betas);
        cudaFree(this->position_of_facility);
        cudaFree(this->facility_at_position);

        // Graph stuff
        cudaFree(this->graph_seeds);
        cudaFree(this->graph_step);
        if (this->graph_ready) {
            cudaGraphExecDestroy(this->graph_exec);
            cudaGraphDestroy(this->cuda_graph);
        }
        cudaStreamDestroy(this->graph_stream);

        // Intermediate buffers
        cudaFree(this->random_uniform);
        cudaFree(this->force_mat);
        cudaFree(this->force_mat_transposed);
        cudaFree(this->energy_deltas);
        cudaFree(this->swap_probs);
        cudaFree(this->swap_targets);

        // One-shot stuff
        cudaFree(this->proposed_targets);
        cudaFree(this->proposed_deltas);
        cudaFree(this->position_owner);
        //cudaFree(this->partials_scratch);

        // Score buffers
        cudaFree(this->replica_score);
        cudaFree(this->best_score_per_replica);
        cudaFree(this->best_score_per_deck);
        cudaFreeHost(this->best_score_host);
        cudaEventDestroy(this->score_event);
        for (int i = 0; i < kThrottleDepth; ++i) {
            cudaEventDestroy(this->manual_throttle[i]);
        }

        cudaFree(this->time_since_swap_beta);

    } else {
        delete[] this->betas;
        delete[] this->position_of_facility;
        delete[] this->facility_at_position;

        // Graph stuff
        delete[] this->graph_seeds;
        delete[] this->graph_step;

        // Intermediate buffers
        delete[] this->random_uniform;
        delete[] this->force_mat;
        delete[] this->force_mat_transposed;
        delete[] this->energy_deltas;
        delete[] this->swap_probs;
        delete[] this->swap_targets;

        // One-shot stuff
        delete[] this->proposed_targets;
        delete[] this->proposed_deltas;
        delete[] this->position_owner;
        //delete[] this->partials_scratch;

        // Score buffers
        delete[] this->replica_score;
        delete[] this->best_score_per_replica;
        delete[] this->best_score_per_deck;
        delete[] this->best_score_overall;

        delete[] this->time_since_swap_beta;
    }
}
