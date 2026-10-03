# PERL: GPU-Accelerated Simulations
GPU-accelerated simulations from PERL

## Running the Code
1. Clone [this repository](https://github.com/PERL-NU/perl-gpu)
2. Run `mkdir build && make` from the repository root
3. Binaries will be generated in `build/bin/`

### Dependencies
At the time of writing, dependencies are bundled with this code in the `include/` directory.

### Notes
- If you are not using an Ada series GPU, you will need to reconfigure `CuRng` in
  `src/lib/random.cu`. See https://docs.nvidia.com/cuda/curanddx/api/description_ops.html
  for details

## Algorithms
- Simulated Annealing
  - Max-Cut / K-Cut
  - QAP
