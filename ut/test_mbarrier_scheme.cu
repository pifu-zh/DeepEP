// Standalone UT to verify monotonic-counter + parity mbarrier scheme
// BEFORE modifying utils.cuh, we validate the algorithm here

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CUDA_CALL(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

__device__ __forceinline__ int get_lane_id() {
    return threadIdx.x % 32;
}

__device__ __forceinline__ bool elect_one_sync() {
    return __activemask() & (1u << __ffs(__activemask()) - 1) & (1u << get_lane_id());
}

// ============ Proposed SM80 mbarrier scheme ============
// uint64_t mbar = [arrive_count (high 32) | current_count (low 32)]
// arrive: atomicAdd low 32 by arrive_count
// wait: parity = (current_count / arrive_count) % 2
// =======================================================

__device__ __forceinline__ void test_mbarrier_init(uint64_t* mbar_ptr, uint32_t arrive_count) {
    *mbar_ptr = static_cast<uint64_t>(arrive_count) << 32;
}

__device__ __forceinline__ void test_mbarrier_arrive(uint64_t* mbar_ptr) {
    uint32_t arrive_count = static_cast<uint32_t>(*mbar_ptr >> 32);
    atomicAdd(reinterpret_cast<uint32_t*>(mbar_ptr), arrive_count);
}

__device__ __forceinline__ void test_mbarrier_arrive_and_expect_tx(uint64_t* mbar_ptr, int num_bytes) {
    uint32_t arrive_count = static_cast<uint32_t>(*mbar_ptr >> 32);
    atomicAdd(reinterpret_cast<uint32_t*>(mbar_ptr), arrive_count);
}

template <bool kWithMultiStages = false>
__device__ __forceinline__ void test_mbarrier_wait(uint64_t* mbar_ptr, uint32_t& phase, int stage_idx = 0) {
    uint32_t arrive_count = static_cast<uint32_t>(*mbar_ptr >> 32);
    if (arrive_count > 0) {
        volatile uint32_t* counter = reinterpret_cast<volatile uint32_t*>(mbar_ptr);
        uint32_t phase_bit = kWithMultiStages ? ((phase >> stage_idx) & 1) : (phase & 1);
        uint32_t expected_parity = phase_bit ^ 1;

        // Spin until parity matches expected
        while (((*counter) / arrive_count) % 2 != expected_parity) {
            __threadfence_block();
        }
        __threadfence_block();
    }
    __syncwarp();
    if constexpr (kWithMultiStages)
        phase ^= (1u << stage_idx);
    else
        phase ^= 1u;
}

__device__ __forceinline__ void test_mbarrier_reset(uint64_t* mbar_ptr) {
    // No-op for monotonic counter scheme
}

// ============ Test Kernels ============

// TC-A: Single producer warp -> single consumer warp, 10000 iterations
__global__ void tcA_single_producer_consumer(int* error_count, int num_iterations) {
    extern __shared__ __align__(16) uint8_t smem[];
    int warp_id = threadIdx.x / 32;
    int lane_id = get_lane_id();

    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem);
    int* shared_val = reinterpret_cast<int*>(smem + sizeof(uint64_t));
    uint32_t phase = 0;

    if (warp_id == 0 && lane_id == 0) {
        *shared_val = 0;
        test_mbarrier_init(mbar, 1);
    }
    __syncthreads();

    for (int iter = 0; iter < num_iterations; iter++) {
        if (warp_id == 0) {
            if (lane_id == 0) *shared_val = iter + 1;
            __syncwarp();
            if (elect_one_sync())
                test_mbarrier_arrive_and_expect_tx(mbar, 0);
        } else if (warp_id == 1) {
            test_mbarrier_wait(mbar, phase);
            __syncwarp();
            if (lane_id == 0) {
                int val = *shared_val;
                if (val != iter + 1) {
                    atomicAdd(error_count, 1);
                }
            }
        }
    }
}

// TC-B: 1 producer warp -> 3 consumer warps, with empty/full barrier handshake
__global__ void tcB_multi_consumer_handshake(int* error_count, int num_iterations) {
    extern __shared__ __align__(16) uint8_t smem[];
    int warp_id = threadIdx.x / 32;
    int lane_id = get_lane_id();

    constexpr int kNumConsumers = 3;
    uint64_t* full_mbar = reinterpret_cast<uint64_t*>(smem);
    uint64_t* empty_mbar = reinterpret_cast<uint64_t*>(smem + sizeof(uint64_t));
    int* shared_val = reinterpret_cast<int*>(smem + sizeof(uint64_t) * 2);

    if (warp_id == 0 && lane_id == 0) {
        *shared_val = 0;
        test_mbarrier_init(full_mbar, 1);
        test_mbarrier_init(empty_mbar, kNumConsumers);
    }
    __syncthreads();

    uint32_t phase = 0;
    if (warp_id == 0) phase = 1;  // Producer starts with phase=1 (expects parity 0 after wait)

    for (int iter = 0; iter < num_iterations; iter++) {
        if (warp_id == 0) {
            // Producer: wait for empty, write data, signal full
            test_mbarrier_wait(empty_mbar, phase);
            __syncwarp();
            if (lane_id == 0) *shared_val = iter + 1;
            __syncwarp();
            if (elect_one_sync())
                test_mbarrier_arrive_and_expect_tx(full_mbar, 0);
        } else if (warp_id >= 1 && warp_id <= kNumConsumers) {
            // Consumer: wait for full, read data, signal empty
            test_mbarrier_wait(full_mbar, phase);
            __syncwarp();
            if (lane_id == 0) {
                int val = *shared_val;
                if (val != iter + 1) {
                    atomicAdd(error_count, 1);
                }
            }
            __syncwarp();
            if (elect_one_sync())
                test_mbarrier_arrive(empty_mbar);
        }
    }
}

// TC-C: Multi-stage pipeline (2 stages) to verify kWithMultiStages
__global__ void tcC_multi_stage(int* error_count, int num_iterations) {
    extern __shared__ __align__(16) uint8_t smem[];
    int warp_id = threadIdx.x / 32;
    int lane_id = get_lane_id();

    constexpr int kNumStages = 2;
    uint64_t* full_mbars[2] = {
        reinterpret_cast<uint64_t*>(smem),
        reinterpret_cast<uint64_t*>(smem + sizeof(uint64_t))
    };
    int* shared_vals[2] = {
        reinterpret_cast<int*>(smem + sizeof(uint64_t) * 2),
        reinterpret_cast<int*>(smem + sizeof(uint64_t) * 2 + sizeof(int))
    };

    if (warp_id == 0 && lane_id == 0) {
        for (int s = 0; s < kNumStages; s++) {
            test_mbarrier_init(full_mbars[s], 1);
            *shared_vals[s] = 0;
        }
    }
    __syncthreads();

    uint32_t phase = 0;

    for (int iter = 0; iter < num_iterations; iter++) {
        int stage = iter % kNumStages;
        if (warp_id == 0) {
            if (lane_id == 0) *shared_vals[stage] = iter + 1;
            __syncwarp();
            if (elect_one_sync())
                test_mbarrier_arrive_and_expect_tx(full_mbars[stage], 0);
        } else if (warp_id == 1) {
            test_mbarrier_wait<true>(full_mbars[stage], phase, stage);
            __syncwarp();
            if (lane_id == 0) {
                int val = *shared_vals[stage];
                if (val != iter + 1) {
                    atomicAdd(error_count, 1);
                }
            }
        }
    }
}

// ============ Test Runner ============

bool run_test(const char* name, void (*kernel)(int*, int), int num_threads, int smem_size, int iterations) {
    int *d_error, h_error;
    CUDA_CALL(cudaMalloc(&d_error, sizeof(int)));
    CUDA_CALL(cudaMemset(d_error, 0, sizeof(int)));

    printf("  Running %s (%d iterations)...\n", name, iterations);
    kernel<<<1, num_threads, smem_size>>>(d_error, iterations);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(&h_error, d_error, sizeof(int), cudaMemcpyDeviceToHost));

    printf("    Errors: %d / %d\n", h_error, iterations);
    CUDA_CALL(cudaFree(d_error));

    if (h_error > 0) {
        printf("  [FAIL] %s\n", name);
        return false;
    }
    printf("  [PASS] %s\n", name);
    return true;
}

int main() {
    int device = 0;
    CUDA_CALL(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, device));
    printf("Running on: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    bool ok = true;
    constexpr int kNumIters = 10000;

    ok &= run_test("TC-A: Single producer -> single consumer",
                   tcA_single_producer_consumer, 64,
                   sizeof(int) + sizeof(uint64_t), kNumIters);

    ok &= run_test("TC-B: 1 producer -> 3 consumers (handshake)",
                   tcB_multi_consumer_handshake, 128,
                   sizeof(int) + sizeof(uint64_t) * 2, kNumIters);

    ok &= run_test("TC-C: Multi-stage (2 stages) pipeline",
                   tcC_multi_stage, 64,
                   sizeof(uint64_t) * 2 + sizeof(int) * 2, kNumIters);

    printf("\n=== %s ===\n", ok ? "ALL PASSED" : "SOME FAILED");
    return ok ? 0 : 1;
}
