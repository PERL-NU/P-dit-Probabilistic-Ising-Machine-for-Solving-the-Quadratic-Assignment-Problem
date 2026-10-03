#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include <atomic>
#include <chrono>
#include <curand_kernel.h>
#include <errno.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <stdio.h>
#include <string>
#include <thread>
#include <vector>


using std::cout;
using std::endl;

const int MAXN = 160; // maximum size of the problem
int N = 100;

const int NUMITERA = 4; // Iterations per mega iteration

int NUMMEGAITERA = 2147483647;

const int MAXNUMREPLICAS = 10; // maximum number of replicas
int NUMREPLICAS = MAXNUMREPLICAS;

const int MAXNUMTHREADS = 32; // maximum number of threads
int NUMTHREADS = MAXNUMTHREADS;

double beta0 = exp2(-12); // largest used beta
// sweep 26 is exp(-12)

double deltaBeta = 1.2; // divisive

double lowerBetaProportion = 0.1;
double higherBetaProportion = 0.9;

bool automaticMode = true; // if true then a series of short trials at different betas are used to try and choose the best beta0 and deltaBeta
bool automaticModeDone = false;

std::thread threads[MAXNUMTHREADS];
std::string inputFileName = "";
std::string outputFolderName = ""; // output folder name

// first is smallest beta, last is largest beta
alignas(64) int indexOfBetaOrder[MAXNUMTHREADS * MAXNUMREPLICAS];
alignas(64) double beta[MAXNUMTHREADS * MAXNUMREPLICAS];
std::mt19937 rng[MAXNUMTHREADS];

int goal = 0;      // goal objective value, program halts when reached
int goalPrint = 0; // goal objective value, prints solution when reached
int timeout = 0;   // timeout in seconds, program halts when reached

alignas(64) int bestE[MAXNUMTHREADS * MAXNUMREPLICAS];
alignas(64) int threadBestE[MAXNUMTHREADS];
alignas(64) int currentE[MAXNUMTHREADS * MAXNUMREPLICAS];
std::atomic<int> singleBestE = 2147483647; // single best energy found

// let location be 0...N-1 for p-dit
alignas(64) int facilityOfLocation[MAXNUMTHREADS * MAXNUMREPLICAS * MAXN];
alignas(64) int locationOfFacility[MAXNUMTHREADS * MAXNUMREPLICAS * MAXN];

// first index is p-dit (location), second is dimension (facility)
alignas(64) int I[MAXNUMTHREADS * MAXNUMREPLICAS * MAXN * MAXN];

// order on https://coral.ise.lehigh.edu/data-sets/qaplib/qaplib-problem-instances-and-solutions/
// is each index is location, and value is facility at that location... just like we specify
alignas(64) int D[MAXN * MAXN];
alignas(64) int F[MAXN * MAXN];

alignas(64) int timeSinceSwap[MAXNUMTHREADS];

int auto_best_found_index[MAXNUMTHREADS];
double auto_best_found_energy[MAXNUMTHREADS];

std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

std::ofstream myfile[MAXNUMTHREADS];
std::ofstream logFile;

static int readProblem() {
    std::ifstream inputFile(inputFileName);

    if (!inputFile.is_open()) {
        std::cerr << "Error: Could not open the file (" << inputFileName << ")." << endl;
        return 1;
    }

    inputFile >> N; // read the size of the matrix
    if (N > MAXN) {
        std::cerr << "Error: N exceeded MAXN (" << MAXN << ")." << endl;
        std::cerr << "Please recompile with a larger MAXN." << endl;
        return 1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            inputFile >> F[i * MAXN + j];
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            inputFile >> D[i * MAXN + j];
        }
    }
    inputFile.close();

    return 0;
}

static void setup() {
    std::random_device dev;
    for (int t = 0; t < NUMTHREADS; t++) {
        timeSinceSwap[t] = 0;
        double runningBeta = beta0;
        for (int l = NUMREPLICAS - 1; l >= 0; l--) {
            indexOfBetaOrder[t * MAXNUMREPLICAS + l] = l;
            beta[t * MAXNUMREPLICAS + l] = runningBeta;
            runningBeta /= deltaBeta; // decrease beta for each replica
        }
    }

    for (int t = 0; t < NUMTHREADS; t++) {
        for (int l = 0; l < NUMREPLICAS; l++) {
            bestE[t * MAXNUMREPLICAS + l] = 2147483647; // initialize best energy to a large value
            currentE[t * MAXNUMREPLICAS + l] = 0;       // initialize current energy to 0
        }
    }

    for (int t = 0; t < NUMTHREADS; t++) {
        threadBestE[t] = 2147483647;
        rng[t] = std::mt19937(dev()); // initialize random number generator
    }

    for (int t = 0; t < NUMTHREADS; t++) {
        for (int l = 0; l < NUMREPLICAS; l++) {
            for (int i = 0; i < N; i++) {
                facilityOfLocation[(t * MAXNUMREPLICAS + l) * MAXN + i] = i;
                locationOfFacility[(t * MAXNUMREPLICAS + l) * MAXN + i] = i;
            }
        }
    }

    if (true || !automaticMode || automaticModeDone) {
        // if not automatic mode, randomly swap around facilities
        for (int t = 0; t < NUMTHREADS; t++) {
            for (int i = 0; i < N; i++) {
                std::uniform_int_distribution<std::mt19937::result_type> distN(0, N - 1); // distribution in range [0, N-1]
                    for (int l = 0; l < NUMREPLICAS; l++) {
                                        int j = distN(rng[0]);                                                    // TODO NOTE THIS WAS ORIGINALLY DONE ON A PER-THREAD LEVEL. NOW NEED PER REPLICA BC BETA SWEEP ITSELF TAKES TOO LONG WITH MANY REPLICA
                        if (i != j) {
                        // swap facilityOfLocation and locationOfFacility
                        if (true || !automaticMode || (automaticModeDone && (t * MAXNUMREPLICAS + l) != auto_best_found_index[t])) {
                            std::swap(facilityOfLocation[(t * MAXNUMREPLICAS + l) * MAXN + i], facilityOfLocation[(t * MAXNUMREPLICAS + l) * MAXN + j]);
                            int f1 = facilityOfLocation[(t * MAXNUMREPLICAS + l) * MAXN + i];
                            int f2 = facilityOfLocation[(t * MAXNUMREPLICAS + l) * MAXN + j];
                            std::swap(locationOfFacility[(t * MAXNUMREPLICAS + l) * MAXN + f1], locationOfFacility[(t * MAXNUMREPLICAS + l) * MAXN + f2]);
                        }
                    }
                }
            }
        }
    }

    for (int t = 0; t < NUMTHREADS; t++) {
        for (int l = 0; l < NUMREPLICAS; l++) {
            for (int i = 0; i < N; i++) {
                // First, initialize the 'a' row to zeros
                for (int a = 0; a < N; a++) {
                    I[((t * MAXNUMREPLICAS + l) * MAXN + i) * MAXN + a] = 0;
                }

                // Then, perform the calculation
                for (int a = 0; a < N; a++) {
                    for (int j = 0; j < N; j++) {
                        I[((t * MAXNUMREPLICAS + l) * MAXN + i) * MAXN + a] -= D[i * MAXN + j] * F[a * MAXN + (facilityOfLocation[(t * MAXNUMREPLICAS + l) * MAXN + j])];
                        I[((t * MAXNUMREPLICAS + l) * MAXN + i) * MAXN + a] -= D[j * MAXN + i] * F[(facilityOfLocation[(t * MAXNUMREPLICAS + l) * MAXN + j]) * MAXN + a];
                    }
                }
            }
        }
    }

    // calculate initial energy
    for (int t = 0; t < NUMTHREADS; t++) {
        for (int l = 0; l < NUMREPLICAS; l++) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    currentE[t * MAXNUMREPLICAS + l] += D[i * MAXN + j] * F[(facilityOfLocation[(t * MAXNUMREPLICAS + l) * MAXN + i]) * MAXN + (facilityOfLocation[(t * MAXNUMREPLICAS + l) * MAXN + j])];
                }
            }
        }
    }
}

// 1. Universal Warp Reduction (Sum)
__inline__ __device__ float warpReduceSum(float val) {
    unsigned int mask = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}

// 2. Universal Warp Inclusive Scan (Running Total)
__inline__ __device__ float warpInclusiveScan(float val) {
    unsigned int mask = __activemask();
    int laneId = threadIdx.x & 31;
    float temp;

#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        temp = __shfl_up_sync(mask, val, offset);
        if (laneId >= offset)
            val += temp;
    }
    return val;
}

__global__ void setup_rand_kernel(curandState *global_rng_state, unsigned long long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x; // necessary to change the id

    int l = blockIdx.x % MAXNUMREPLICAS; // 0 to 9
    int t = blockIdx.x / MAXNUMREPLICAS; // Which independent instance this is
    int b = threadIdx.x;                 // Thread handles one specific 'b' element
    if (b == 0) {
        curand_init(seed, id, 0, &global_rng_state[t * MAXNUMREPLICAS + l]);
    }
}

__global__ void persistent_sim_kernel(
    int *__restrict__ facilityOfLocation,
    int *__restrict__ locationOfFacility,
    int *__restrict__ I,
    const int *__restrict__ D,
    const int *__restrict__ F,
    double *__restrict__ beta,
    int *__restrict__ bestE,
    int *__restrict__ currentE,
    int *__restrict__ indexOfBetaOrder,
    curandState *global_rng_state,
    int *singleBest,
    int *killStatus,
    int megaitera,
    int N) {
    // Map hardware to your hierarchy
    int l = blockIdx.x % MAXNUMREPLICAS; // 0 to 9
    int t = blockIdx.x / MAXNUMREPLICAS; // Which independent instance this is
    int b = threadIdx.x;                 // Thread handles one specific 'b' element

    int warpId = b / 32;
    int laneId = b % 32;

    // Setup replica-specific float constant

    int m = 0;

    int timeSinceSwap = 0;
    while (m < megaitera) {
        if (l == 0 && b == 0) {
            timeSinceSwap++;
            for (int le = 0; le < MAXNUMREPLICAS - 1; le++) {
                int index1 = indexOfBetaOrder[t * MAXNUMREPLICAS + le];
                int index2 = indexOfBetaOrder[t * MAXNUMREPLICAS + (le + 1)];
                int e1 = currentE[t * MAXNUMREPLICAS + index1];
                int e2 = currentE[t * MAXNUMREPLICAS + index2];

                if (e1 < e2) {
                    indexOfBetaOrder[t * MAXNUMREPLICAS + le] = index2;
                    indexOfBetaOrder[t * MAXNUMREPLICAS + (le + 1)] = index1;
                    double temp = beta[t * MAXNUMREPLICAS + index1];
                    beta[t * MAXNUMREPLICAS + index1] = beta[t * MAXNUMREPLICAS + index2];
                    beta[t * MAXNUMREPLICAS + index2] = temp;
                    if (index2 == MAXNUMREPLICAS - 1) {
                        timeSinceSwap = 0;
                    }
                }
            }

            if (timeSinceSwap > 100) {
                int index1 = indexOfBetaOrder[t * MAXNUMREPLICAS + MAXNUMREPLICAS - 1];
                int index2 = indexOfBetaOrder[t * MAXNUMREPLICAS + 0];

                indexOfBetaOrder[t * MAXNUMREPLICAS + MAXNUMREPLICAS - 1] = index2;
                indexOfBetaOrder[t * MAXNUMREPLICAS + 0] = index1;

                double temp = beta[t * MAXNUMREPLICAS + index1];
                beta[t * MAXNUMREPLICAS + index1] = beta[t * MAXNUMREPLICAS + index2];
                beta[t * MAXNUMREPLICAS + index2] = temp;
            }
        }

        int iteration = 0;
        int offset = (t * MAXNUMREPLICAS + l) * MAXN;

        __syncthreads();

        float this_beta = beta[t * MAXNUMREPLICAS + l]; // Example specific constant

        while (iteration < NUMITERA) {
            int i = 0;
            while (i < N) {
                __shared__ float r;
                __shared__ int winning_b;
                __shared__ int winning_j;
                __shared__ float shared_comm[32];
                __shared__ float probSum;

                if (b == 0) {
                    winning_b = -1;
                    curandState local_state = global_rng_state[t * MAXNUMREPLICAS + l];

                    r = curand_uniform(&local_state);

                    global_rng_state[t * MAXNUMREPLICAS + l] = local_state;
                }
                if (b < 32) {
                    shared_comm[b] = 0.0f;
                }

                int offset_i = (offset + i) * MAXN;
                int deltaE = 0;
                float probs = 0;
                int a = 0;
                int j = 0;
                int offset_j = 0;

                if (b < N) {
                    a = facilityOfLocation[offset + i];

                    j = locationOfFacility[offset + b]; // reverse lookup
                    offset_j = (offset + j) * MAXN;
                    deltaE = (I[offset_i + b] + I[offset_j + a] - I[offset_i + a] - I[offset_j + b] - (D[i * MAXN + i] + D[j * MAXN + j] - D[i * MAXN + j] - D[j * MAXN + i]) * (F[a * MAXN + a] + F[b * MAXN + b] - F[a * MAXN + b] - F[b * MAXN + a]));

                    float expr = this_beta * deltaE;
                    probs = exp2(expr);
                }


                // --- PHASE 1: THE GRAND SUM ---
                float my_val = (b < N) ? probs : 0.0f;
                float sum = warpReduceSum(my_val);

                // Each warp writes its total to shared memory
                if (laneId == 0)
                    shared_comm[warpId] = sum;

                __syncthreads();

                // First warp sums the warp totals
                if (warpId == 0) {
                    float total = (b < 32) ? shared_comm[b] : 0.0f;
                    total = warpReduceSum(total);
                    if (b == 0)
                        probSum = total; // Store the final total for everyone
                }
                __syncthreads();

                // --- PHASE 2: THE INCLUSIVE SCAN & THRESHOLD ---
                float scaledR = probSum * r;

                // A. Initial warp-level scan
                float my_running_sum = warpInclusiveScan(my_val);

                // B. Handle the carries between warps
                // We reuse shared_comm to store warp-level totals for the scan
                if (laneId == 31 || b == N - 1)
                    shared_comm[warpId] = my_running_sum;
                __syncthreads();

                if (warpId == 0) {
                    float c = (b < 32) ? shared_comm[b] : 0.0f;
                    c = warpInclusiveScan(c);
                    if (b < 32)
                        shared_comm[b] = c;
                }
                __syncthreads();

                // C. Add the carry from the PREVIOUS warp
                if (warpId > 0) {
                    my_running_sum += shared_comm[warpId - 1];
                }

                // Logic to prevent floating point weirdness from causing two different b's to trigger. This method only fails if all of 32 b's between warps are ~0 and the threshold is there
                bool i_crossed = my_running_sum >= scaledR && b < N && probs > 0;
                unsigned int mask = __activemask();
                unsigned int ballot = __ballot_sync(mask, i_crossed);

                if (laneId == 0) {
                    shared_comm[warpId] = (ballot != 0) ? 1 : 0;
                }
                __syncthreads();

                if (i_crossed) {
                    int leader_lane = __ffs(ballot) - 1;

                    if (laneId == leader_lane) {
                        // Check if ANY warp before us had a crosser
                        bool earlier_warp_crossed = false;
                        for (int w = 0; w < 5; w++) {
                            if (shared_comm[w] == 1 && w < warpId) {
                                earlier_warp_crossed = true;
                            }
                        }

                        // If we are the leader of our warp AND no warp before us crossed...
                        if (!earlier_warp_crossed) {
                            if (b != a) {
                                winning_b = b;

                                winning_j = locationOfFacility[offset + b]; // reverse lookup
                                currentE[t * MAXNUMREPLICAS + l] -= deltaE;

                                if (currentE[t * MAXNUMREPLICAS + l] < bestE[t * MAXNUMREPLICAS + l]) {
                                    bestE[t * MAXNUMREPLICAS + l] = currentE[t * MAXNUMREPLICAS + l];

                                    atomicMin(singleBest, currentE[t * MAXNUMREPLICAS + l]);
                                }

                                facilityOfLocation[offset + i] = b; // swap facility of location i with b
                                locationOfFacility[offset + b] = i; // update reverse lookup

                                facilityOfLocation[offset + winning_j] = a; // swap facility of location j with a
                                locationOfFacility[offset + a] = winning_j; // update reverse lookup
                            }
                        }
                    }
                }

                __syncthreads();

                if (winning_b != -1) {
                    for (int k = 0; k < N; k++) {
                        int delta = D[k * MAXN + winning_j] - D[k * MAXN + i];
                        int delta2 = D[winning_j * MAXN + k] - D[i * MAXN + k];
                        // c became b, b became winning_b
                        if (b < N) {
                            I[(offset + k) * MAXN + b] += delta * (F[b * MAXN + winning_b] - F[b * MAXN + a]) + delta2 * (F[winning_b * MAXN + b] - F[a * MAXN + b]);
                        }
                    }
                }

                __syncthreads();

                i++;
            }

            iteration++;
        }
        m = ((*killStatus) ? 2147483647 : m + 1);
    }
}

void doMegaItera(int t, int l) {
    std::uniform_int_distribution<std::mt19937::result_type> distN(0, N - 1); // distribution in range [0, N-1]
    long double probs[MAXN];
    int deltaEs[MAXN];
    double scaledR = 0.0;
    double cumulativeProb = 0.0;

    double this_beta = beta[t * MAXNUMREPLICAS + l];

    int offset = (t * MAXNUMREPLICAS + l) * MAXN;

    for (int itera = 0; itera < NUMITERA; itera++) {
        for (int i = 0; i < N; i++) {
            int maxDeltaE = 0;

            int offset_i = (offset + i) * MAXN;

            double r = std::generate_canonical<double, std::numeric_limits<double>::digits>(rng[t]);

            double probSum = 0.0f;
            int a = facilityOfLocation[offset + i]; // forward lookup

            // calculate probabilities
            // assume i: a -> b... j: b-> a

            for (int b = 0; b < N; b++) {
                int j = locationOfFacility[offset + b]; // reverse lookup
                int offset_j = (offset + j) * MAXN;

                // factor of 2 is only because original problem doesnt have 1/2 factor for asymetric connections so ab ba both count as "1" full
                int deltaE = (I[offset_i + b] + I[offset_j + a] - I[offset_i + a] - I[offset_j + b] - (D[i * MAXN + i] + D[j * MAXN + j] - D[i * MAXN + j] - D[j * MAXN + i]) * (F[a * MAXN + a] + F[b * MAXN + b] - F[a * MAXN + b] - F[b * MAXN + a]));

                deltaEs[b] = deltaE;

                if (deltaE > maxDeltaE) {
                    maxDeltaE = deltaE;
                }
            }

            double low_cutoff_double = maxDeltaE - 20.0 / this_beta;
            int low_cutoff_int = static_cast<int>(low_cutoff_double);
            for (int b = 0; b < N; b++) {
                probs[b] = 0.0;

                if (deltaEs[b] > low_cutoff_int) {
                    long double expr = this_beta * deltaEs[b];

                    probs[b] = exp2(expr);
                    probSum += probs[b];
                }
            }

            scaledR = r * probSum;
            cumulativeProb = 0.0;
            for (int b = 0; b < N; b++) {
                cumulativeProb += probs[b];
                if (cumulativeProb >= scaledR) {
                    if (b != a) {
                        int j = locationOfFacility[offset + b]; // reverse lookup
                        int offset_j = (offset + j) * MAXN;

                        currentE[t * MAXNUMREPLICAS + l] -= deltaEs[b];
                        facilityOfLocation[offset + i] = b; // swap facility of location i with b
                        locationOfFacility[offset + b] = i; // update reverse lookup

                        facilityOfLocation[offset + j] = a; // swap facility of location j with a
                        locationOfFacility[offset + a] = j; // update reverse lookup

                        for (int k = 0; k < N; k++) {
                            int delta = D[k * MAXN + j] - D[k * MAXN + i];
                            int delta2 = D[j * MAXN + k] - D[i * MAXN + k];
                            for (int c = 0; c < N; c++) {
                                I[(offset + k) * MAXN + c] = I[(offset + k) * MAXN + c] + delta * (F[c * MAXN + b] - F[c * MAXN + a]) + delta2 * (F[b * MAXN + c] - F[a * MAXN + c]);
                            }
                        }

                        if (currentE[t * MAXNUMREPLICAS + l] < bestE[t * MAXNUMREPLICAS + l]) {
                            bestE[t * MAXNUMREPLICAS + l] = currentE[t * MAXNUMREPLICAS + l];
                            if (goalPrint > 0 && bestE[t * MAXNUMREPLICAS + l] < goalPrint) {
                                logFile << "Thread " << t << " Replica " << l << ": Goal surpassed: " << bestE[t * MAXNUMREPLICAS + l] << endl;
                                cout << "Thread " << t << " Replica " << l << ": Goal surpassed: " << bestE[t * MAXNUMREPLICAS + l] << endl;
                                for (int i = 0; i < N; i++) {
                                    logFile << facilityOfLocation[offset + i] << " ";
                                    cout << facilityOfLocation[offset + i] << " ";
                                }
                                logFile << endl;
                                cout << endl;
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
}

void threadToRun(int t) {
    for (int m = 0; m < NUMMEGAITERA; m++) {
        bool foundBetter = false;
        timeSinceSwap[t]++;

        for (int l = 0; l < NUMREPLICAS; l++) {
            if (bestE[t * MAXNUMREPLICAS + l] < threadBestE[t]) {
                foundBetter = true;
                threadBestE[t] = bestE[t * MAXNUMREPLICAS + l];
                timeSinceSwap[t] = 0;

                if (threadBestE[t] < singleBestE) {
                    singleBestE = threadBestE[t];
                    auto now = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();

                    logFile << "Best Energy Found: " << singleBestE << " Time (s): " << elapsed / 1000.0 << endl;
                    cout << "Best Energy Found: " << singleBestE << " Time (s): " << elapsed / 1000.0 << endl;
                }
            }
        }

        if (foundBetter) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - startTime);

            myfile[t] << m << " " << duration.count() << " " << threadBestE[t] << endl;
        }

        //  loop from smallest beta to largest beta
        for (int l = 0; l < NUMREPLICAS - 1; l++) {
            int index1 = indexOfBetaOrder[t * MAXNUMREPLICAS + l];
            int index2 = indexOfBetaOrder[t * MAXNUMREPLICAS + (l + 1)];
            double e1 = currentE[t * MAXNUMREPLICAS + index1];
            double e2 = currentE[t * MAXNUMREPLICAS + index2];

            if (e1 < e2) {
                std::swap(indexOfBetaOrder[t * MAXNUMREPLICAS + l], indexOfBetaOrder[t * MAXNUMREPLICAS + (l + 1)]);
                std::swap(beta[t * MAXNUMREPLICAS + index1], beta[t * MAXNUMREPLICAS + index2]);
                if (index2 == NUMREPLICAS - 1) {
                    timeSinceSwap[t] = 0;
                }
            }
        }

        if (timeSinceSwap[t] > 100) {
            int index1 = indexOfBetaOrder[t * MAXNUMREPLICAS + MAXNUMREPLICAS - 1];
            int index2 = indexOfBetaOrder[t * MAXNUMREPLICAS + 0];
            std::swap(indexOfBetaOrder[t * MAXNUMREPLICAS + MAXNUMREPLICAS - 1], indexOfBetaOrder[t * MAXNUMREPLICAS + 0]);
            std::swap(beta[t * MAXNUMREPLICAS + index1], beta[t * MAXNUMREPLICAS + index2]);
        }

        if (singleBestE <= goal) {
            return;
        }

        if (timeout > 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();

            if (elapsed >= timeout) {
                return;
            }
        }

        // at end so can swap after initial autoSelectParameters
        for (int l = 0; l < NUMREPLICAS; l++) {
            doMegaItera(t, l);
        }
    }
}

const double minBetaToTry = -30;  // minimum beta to try in automatic mode
const double maxBetaToTry = 1;    // maximum beta to try in automatic mode
const int numTrials = 1;          // number of trials for each beta for each thread in automatic mode
const double marginFactor = 0.05; // margin factor for determining the best beta
const double betaStep = 1;        // step size for exp2(beta) in automatic mode
const int numSteps = (maxBetaToTry - minBetaToTry) / betaStep + 1;

static void autoSelectParameters() {
    double minBeta = 0;
    double maxBeta = 0;

    double baseLineResult = 0;

    double lastEnergySum = 0;

    int tempNUMMEGAITERA = NUMMEGAITERA;
    double tempDeltaBeta = deltaBeta;

    int tempNUMREPLICAS = NUMREPLICAS;

    // magic number for now
    double bestEnergySums[1000] = {0};

    deltaBeta = 1; // have all trials at same beta

    NUMMEGAITERA = 10;

    bool hasStartedDecrease = false;

    for (int i = 0; i < numSteps; i++) {
        double b = minBetaToTry + i * betaStep; // calculate beta for this step
        beta0 = exp2(b);

        for (int trial = 0; trial < numTrials; trial++) {
            setup();
            for (int t = 0; t < NUMTHREADS; t++) {
                threads[t] = std::thread([t]() {
                    threadToRun(t);
                });
            }

            for (int t = 0; t < NUMTHREADS; t++) {
                threads[t].join();
                auto_best_found_index[t] = 0;
                auto_best_found_energy[t] = 1e30;
                for (int l = 0; l < NUMREPLICAS; l++) {
                    bestEnergySums[i] += currentE[t * MAXNUMREPLICAS + l] / (static_cast<double>(numTrials) * NUMTHREADS * NUMREPLICAS);
                    if (currentE[t * MAXNUMREPLICAS + l] < auto_best_found_energy[t]) {
                        auto_best_found_energy[t] = currentE[t * MAXNUMREPLICAS + l];
                        auto_best_found_index[t] = t * MAXNUMREPLICAS + l;
                    }
                }
            }
            // check to stop early if solution found during sweep
            if (singleBestE <= goal) {
                return;
            }
        }

        logFile << "Beta: " << exp2(b) << ", Average Energy: " << bestEnergySums[i] << endl;
        cout << "Beta: " << exp2(b) << ", Average Energy: " << bestEnergySums[i] << endl;

        // magic number for now
        if (i > 0 && bestEnergySums[i] < 0.9 * bestEnergySums[i - 1]) {
            hasStartedDecrease = true; // we have started to decrease
        }

        // we have flattened out completely or have started to rise again, so we can stop
        if ((i == numSteps - 1)) {
            double minE = 1e30;
            int minEIndex = 0;
            double maxE = -1e30;
            for (int j = 0; j < i; j++) {
                if (bestEnergySums[j] < minE) {
                    minE = bestEnergySums[j];
                    minEIndex = j;
                }
                if (j == 0) { // Assume random noise is "max" in case we actually trail up towards the end past the min
                    maxE = bestEnergySums[j];
                }
            }
            for (int j = 0; j < i; j++) {
                if (minBeta == 0 && ((j == (minEIndex - 1) && (j >= 1)) || (bestEnergySums[j] < (maxE - minE) * higherBetaProportion + minE))) {
                    minBeta = exp2(minBetaToTry + (j)*betaStep);
                    logFile << "Selecting minimum beta of " << (minBeta) << endl;
                    cout << "Selecting minimum beta of " << (minBeta) << endl;
                }
                if (j == i - 1 || (minBeta != 0 && (j == minEIndex || (bestEnergySums[j] < (maxE - minE) * lowerBetaProportion + minE)))) {
                    if (minBeta == 0) {
                        minBeta = exp2(minBetaToTry + 0 * betaStep);
                    }
                    maxBeta = exp2(minBetaToTry + (j + 2) * betaStep); // TODO THE PLUS TWO CAN BE MADE A PARAMETER
                    logFile << "Selecting maximum beta of " << (maxBeta) << endl;
                    cout << "Selecting maximum beta of " << (maxBeta) << endl;

                    beta0 = maxBeta;                                       // set the beta0 to the maximum beta
                    deltaBeta = pow(maxBeta / minBeta, 1.0 / NUMREPLICAS); // calculate deltaBeta based on the ratio of max and min beta
                    NUMMEGAITERA = tempNUMMEGAITERA;                       // restore the number of mega iterations
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - startTime);
                    logFile << "Selecting Beta Parameters took " << duration.count() << " seconds." << endl;
                    cout << "Selecting Beta Parameters took " << duration.count() << " seconds." << endl
                         << endl;
                    automaticModeDone = true;

                    NUMTHREADS = MAXNUMTHREADS; // RESET for GPU

                    setup();
                    return;
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    singleBestE = 2147483647;

    // Iterate through the command-line arguments
    for (int i = 1; i < argc; ++i) { // Start from i=1 to skip the program name
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            cout << "Usage: " << argv[0] << " [OPTIONS]... [FILE]..." << endl;
            cout << "Tries to find solution to provided Quadratic Assignment Problem (QAP) file" << endl;
            cout << endl;
            cout << "-b              set the beta of the largest beta replica" << endl;
            cout << "-g              set the goal objective value, program halts when reached" << endl;
            cout << "-G              set the goal objective value, prints solution when reached" << endl;
            cout << "-h, --help      displays this help message and exits" << endl;
            cout << "-i              set number of iterations between replica swaps, defaults to 4" << endl;
            cout << "-m              set the multiplative difference in beta replicas" << endl;
            cout << "-o              set the output folder name" << endl;
            cout << "-r              set number of repulicas, defaults to maximum compiled for" << endl;
            cout << "-s              set the timeout in seconds, program halts when reached" << endl;
            cout << "-t              set number of threads, defaults to maximum compiled for" << endl;
            cout << "-w              set number of swaps, program halts when reached" << endl;

            cout << endl;
            cout << "If -b is not set, then automatic mode is used." << endl;
            cout << "In automatic mode, a series of short trials at different betas are used to try and choose the best parameters." << endl;
            return 0;
        }

        // todo fix
        if (arg == "-al") {
            if (i + 1 < argc) { // Check if there is a next argument
                lowerBetaProportion = std::stod(argv[++i]); // Convert the next argument to double
            } else {
                std::cerr << "Error: -b option requires a value." << endl;
                return 1;
            }
        }

        if (arg == "-ah") {
            if (i + 1 < argc) { // Check if there is a next argument
                higherBetaProportion = std::stod(argv[++i]); // Convert the next argument to double
            } else {
                std::cerr << "Error: -b option requires a value." << endl;
                return 1;
            }
        }

        if (arg == "-b") {
            if (i + 1 < argc) { // Check if there is a next argument
                // 1.44269504089 is 1/ln(2), due to exp2 being used in the code instead of exp
                beta0 = std::stod(argv[++i]) * 1.44269504089; // Convert the next argument to double
                automaticMode = false;
            } else {
                std::cerr << "Error: -b option requires a value." << endl;
                return 1;
            }
        } else if (arg == "-g") {
            if (i + 1 < argc) {              // Check if there is a next argument
                goal = std::stoi(argv[++i]); // Convert the next argument to double
            } else {
                std::cerr << "Error: -g option requires a value." << endl;
                return 1;
            }
        } else if (arg == "-G") {
            if (i + 1 < argc) {                   // Check if there is a next argument
                goalPrint = std::stoi(argv[++i]); // Convert the next argument to double
            } else {
                std::cerr << "Error: -G option requires a value." << endl;
                return 1;
            }
        } else if (arg == "-i") {
            if (i + 1 < argc) { // Check if there is a next argument
                // NUMITERA = std::stoi(argv[++i]); // Convert the next argument to int // TODO REMOVE OPTION
            } else {
                std::cerr << "Error: -i option requires a value." << endl;
                return 1;
            }
        } else if (arg == "-m") {
            if (i + 1 < argc) {                   // Check if there is a next argument
                deltaBeta = std::stod(argv[++i]); // Convert the next argument to double
                // automaticMode = false;
            } else {
                std::cerr << "Error: -m option requires a value." << endl;
                return 1;
            }
        } else if (arg == "-o") {
            if (i + 1 < argc) {                 // Check if there is a next argument
                outputFolderName = (argv[++i]); // Convert the next argument to double
            } else {
                std::cerr << "Error: -o option requires a value." << endl;
                return 1;
            }
        } else if (arg == "-r") {
            if (i + 1 < argc) {                     // Check if there is a next argument
                NUMREPLICAS = std::stoi(argv[++i]); // Convert the next argument to int
                if (NUMREPLICAS > MAXNUMREPLICAS) {
                    std::cerr << "Error: -r option exceeds maximum number of replicas (" << MAXNUMREPLICAS << ")." << endl;
                    std::cerr << "Please recompile with a larger MAXNUMREPLICAS." << endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: -r option requires a value." << endl;
                return 1;
            }
        } else if (arg == "-s") {
            if (i + 1 < argc) {                 // Check if there is a next argument
                timeout = std::stoi(argv[++i]); // Convert the next argument to int
            } else {
                std::cerr << "Error: -s option requires a value." << endl;
                return 1;
            }
        } else if (arg == "-t") {
            if (i + 1 < argc) { // Check if there is a next argument
                NUMTHREADS = std::stoi(argv[++i]);
                if (NUMTHREADS > MAXNUMTHREADS) {
                    std::cerr << "Error: -t option exceeds maximum number of threads (" << MAXNUMREPLICAS << ")." << endl;
                    std::cerr << "Please recompile with a larger MAXNUMREPLICAS." << endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: -t option requires a value." << endl;
                return 1;
            }
        } else if (arg == "-w") {
            if (i + 1 < argc) {                      // Check if there is a next argument
                NUMMEGAITERA = std::stoi(argv[++i]); // Convert the next argument to int
            } else {
                std::cerr << "Error: -w option requires a value." << endl;
                return 1;
            }
        } else {
            inputFileName = arg; // Store the file name
        }
    }

    if (inputFileName.empty()) {
        std::cerr << "Error: No input file specified." << endl;
        return 1;
    }

    if (outputFolderName.empty()) {
        outputFolderName = inputFileName + ".out";
    }

	namespace fs = std::filesystem;

	// ...

	std::error_code ec;

	// create_directory returns true if it CREATED the directory.
	// It returns false if it failed OR if the directory already exists.
	if (fs::create_directory(outputFolderName, ec)) {
	    // Folder created successfully
	} else {
	    // Check if it failed because it already exists
	    if (fs::exists(outputFolderName)) {
		std::cerr << "Error: Folder '" << outputFolderName << "' already exists." << std::endl;
	    } else {
		// An actual system error occurred (e.g., permission denied)
		std::cerr << "Error: Error creating folder. System error code: " << ec.value() << std::endl;
		return 1;
	    }
	}

    if (readProblem() != 0) {
        return 1;
    }

    setup();

    startTime = std::chrono::high_resolution_clock::now();

    logFile.open(outputFolderName + "/log.txt");
    logFile << "Iterations between swaps " << NUMITERA << endl
            << "Number of Swaps " << NUMMEGAITERA << endl
            << "Number of Replicas " << NUMREPLICAS << endl
            << "Number of Threads " << NUMTHREADS << endl;
    if (!automaticMode) {
        logFile << "Maximum Beta " << beta0 << endl
                << "Multiplicative Beta between iterations " << deltaBeta << endl;
        cout << "Maximum Beta " << beta0 << endl
             << "Multiplicative Beta between iterations " << deltaBeta << endl;
    } else {
        logFile << "Automatic mode enabled, Beta parameters will be determined automatically." << endl;
        autoSelectParameters();
        automaticMode = false;
        logFile << "Maximum Beta " << beta0 << endl
                << "Multiplicative Beta between iterations " << deltaBeta << endl;
        cout << "Maximum Beta " << beta0 << endl
             << "Multiplicative Beta between iterations " << deltaBeta << endl;
    }
    logFile << "Input file: " << inputFileName << endl;

    if (!(singleBestE <= goal)) {
        int num_instances = MAXNUMTHREADS;
        int num_replicas = MAXNUMREPLICAS;
        int total_blocks = num_instances * num_replicas;
        int threads_per_block = ((N + 31) / 32) * 32;

        // Pointers for Pinned Memory (Accessible by both CPU and GPU)
        int *d_singleBest;
        int *d_kill_signal;

        // 1. Allocate Pinned Memory
        cudaMallocHost((void **)&d_singleBest, sizeof(int));
        cudaMallocHost((void **)&d_kill_signal, sizeof(int));

        // 2. Initialize Default Values
        *d_singleBest = 2147483647; // Default: nothing found yet
        *d_kill_signal = 0;         // Default: don't stops

        curandState *d_rand_states;
        cudaMalloc((void **)&d_rand_states, MAXNUMTHREADS * MAXNUMREPLICAS * sizeof(curandState));
        // 1. Get the current time point

        auto now2 = std::chrono::high_resolution_clock::now();
        auto duration2 = now2.time_since_epoch();
        long long nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration2).count();

        unsigned long long seed = 2334435ULL + nanoseconds;
        setup_rand_kernel<<<total_blocks, threads_per_block>>>(d_rand_states, seed);

        cudaDeviceSynchronize();

        // 1. Create empty pointers that will hold the GPU memory addresses
        int *d_facilityOfLocation = nullptr;
        int *d_locationOfFacility = nullptr;

        int *d_I = nullptr;
        int *d_D = nullptr;
        int *d_F = nullptr;

        double *d_beta = nullptr;

        int *d_bestE = nullptr;
        int *d_currentE = nullptr;

        int *d_indexOfBetaOrder = nullptr;

        // 2. Allocate VRAM on the RTX 4070
        // cudaMalloc takes a pointer to a pointer, and the size in BYTES
        cudaError_t errfacilityOfLocation = cudaMalloc((void **)&d_facilityOfLocation, MAXNUMTHREADS * MAXNUMREPLICAS * MAXN * sizeof(int));
        cudaError_t errlocationOfFacility = cudaMalloc((void **)&d_locationOfFacility, MAXNUMTHREADS * MAXNUMREPLICAS * MAXN * sizeof(int));

        cudaError_t errI = cudaMalloc((void **)&d_I, MAXNUMTHREADS * MAXNUMREPLICAS * MAXN * MAXN * sizeof(int));

        cudaError_t errD = cudaMalloc((void **)&d_D, MAXN * MAXN * sizeof(int));
        cudaError_t errF = cudaMalloc((void **)&d_F, MAXN * MAXN * sizeof(int));

        cudaError_t errBeta = cudaMalloc((void **)&d_beta, MAXNUMTHREADS * MAXNUMREPLICAS * sizeof(double));

        cudaError_t errBestE = cudaMalloc((void **)&d_bestE, MAXNUMTHREADS * MAXNUMREPLICAS * sizeof(int));
        cudaError_t errCurrentE = cudaMalloc((void **)&d_currentE, MAXNUMTHREADS * MAXNUMREPLICAS * sizeof(int));

        cudaError_t errIndexOfBetaOrder = cudaMalloc((void **)&d_indexOfBetaOrder, MAXNUMTHREADS * MAXNUMREPLICAS * sizeof(int));

        if (errfacilityOfLocation != cudaSuccess || errlocationOfFacility != cudaSuccess || errI != cudaSuccess || errD != cudaSuccess || errF != cudaSuccess || errBeta != cudaSuccess || errBestE != cudaSuccess || errCurrentE != cudaSuccess || errIndexOfBetaOrder != cudaSuccess) {
            std::cerr << "VRAM Allocation Failed!" << std::endl;
            return -1;
        }

        // 3. Copy the data from Host (Motherboard) to Device (GPU)
        // Syntax: (Destination, Source, Size in Bytes, Direction)
        cudaMemcpy(d_facilityOfLocation, facilityOfLocation, MAXNUMTHREADS * MAXNUMREPLICAS * MAXN * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_locationOfFacility, locationOfFacility, MAXNUMTHREADS * MAXNUMREPLICAS * MAXN * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_I, I, MAXNUMTHREADS * MAXNUMREPLICAS * MAXN * MAXN * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_D, D, MAXN * MAXN * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_F, F, MAXN * MAXN * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, beta, MAXNUMTHREADS * MAXNUMREPLICAS * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bestE, bestE, MAXNUMTHREADS * MAXNUMREPLICAS * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_currentE, currentE, MAXNUMTHREADS * MAXNUMREPLICAS * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_indexOfBetaOrder, indexOfBetaOrder, MAXNUMTHREADS * MAXNUMREPLICAS * sizeof(int), cudaMemcpyHostToDevice);

        std::chrono::time_point<std::chrono::high_resolution_clock> startTime2 = std::chrono::high_resolution_clock::now();

        cout << "Starting GPU Simulation..." << endl;

        persistent_sim_kernel<<<total_blocks, threads_per_block>>>(
            d_facilityOfLocation, d_locationOfFacility, d_I, d_D, d_F, d_beta, d_bestE, d_currentE, d_indexOfBetaOrder, d_rand_states, d_singleBest, d_kill_signal, NUMMEGAITERA, N);

        auto end2 = std::chrono::high_resolution_clock::now();

        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            std::cerr << "GPU Launch Error: " << cudaGetErrorString(launch_err) << std::endl;
        }

        while (true) {
            auto now_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - startTime).count();

            // 1. Check if GPU signaled success
            if (*d_singleBest < singleBestE) {
                singleBestE = *d_singleBest;
                cout << "Best Energy Found: " << singleBestE << " Time (s): " << elapsed / 1000.0 << endl;
                logFile << "Best Energy Found: " << singleBestE << " Time (s): " << elapsed / 1000.0 << endl;
            }

            if (singleBestE <= goal || elapsed / 1000.0 > timeout) {
                *d_kill_signal = 1;
                break;
            }

            // 3. Wait 1ms to prevent the CPU from 100% usage while polling
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            std::cerr << "GPU Execution Error: " << cudaGetErrorString(sync_err) << std::endl;
        }

        cout << "Ending GPU Simulation..." << endl;
    }

    // cout << host_log_buffer << endl; // Example of reading back a log value from the GPU

    if (!(singleBestE <= goal)) {
        // for (int t = 0; t < NUMTHREADS; t++) {
        //     myfile[t].open(outputFolderName + "/sweep" + std::to_string(t) + ".txt");
        // }

        // for (int t = 0; t < NUMTHREADS; t++) {
        //     threads[t] = std::thread([t]() {
        //         threadToRun(t);
        //         });
        // }

        // for (int t = 0; t < NUMTHREADS; t++) {
        //     threads[t].join();
        // }
    }
    cout << endl;
    cout << "Halting program." << endl;
    logFile << "Single best energy found: " << singleBestE << endl;
    cout << "Single best energy found: " << singleBestE << endl;
    cout << "AFD " << (1.0 * singleBestE - goal) / goal * 100.0 << endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - startTime);
    logFile << "Elapsed time: " << duration.count() / 1000.0 << " seconds." << endl;
    cout << "Elapsed time: " << duration.count() / 1000.0 << " seconds." << endl;
}
