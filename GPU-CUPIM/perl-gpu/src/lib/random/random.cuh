#ifndef PERL_RANDOM_CUH
#define PERL_RANDOM_CUH

#include "cuda_runtime.h"
#include <curanddx.hpp>
#include <stdint.h>

/**
 * @brief Random number generator
 *
 * Currently configured to generate Philox4 (vectorized Philox) for the Ada
 * Lovelace architecture. You will need to reconfigure this for other GPUs.
 *
 * Philox outperforms XORWOW, but only when vectorized.
 */
using CuRng = decltype(
    curanddx::Generator<curanddx::philox4_32>()
    + curanddx::SM<890>()  // Ada
    + curanddx::Thread()
);

/**
 * @brief Populate a buffer with random numbers
 *
 * @param n Number of random numbers to generate. MUST be a multiple of 4
 * @param seed Pointer to seed for RNG
 * @param offset RNG seed offset
 * @param buf_d Output buffer
 */
__global__
void generate_random_numbers(
    const uint32_t n,
    const uint64_t *seed,
    const uint64_t *offset,
    float4 *buf_d
);

__global__
void advance_rng(uint64_t *offset, const uint64_t stride);

#endif  /* PERL_RANDOM_CUH */
