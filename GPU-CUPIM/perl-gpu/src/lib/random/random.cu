#include "random.cuh"

#include "cuda_runtime.h"

__global__
void generate_random_numbers(
    const uint32_t n,
    const uint64_t *seed,
    const uint64_t *offset,
    float4 *buf_d
) {
    const uint32_t buf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (4 * buf_idx >= n) {
        return;
    }

    const uint64_t x = *offset + buf_idx;
    CuRng rng(*seed, x % 65536, x / 65536);
    curanddx::uniform<float> distr;
    buf_d[buf_idx] = distr.generate4(rng);
}

__global__
void advance_rng(uint64_t *offset, const uint64_t stride) {
    if (threadIdx.x == 0) {
        *offset += stride;
    }
}
