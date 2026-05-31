#include "../csrc/kernels/utils.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace deep_ep;

#define CUDA_CALL(cmd) do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

#define TEST_PASS() printf("  [PASS] %s\n", __FUNCTION__)
#define TEST_FAIL(msg) do { printf("  [FAIL] %s: %s\n", __FUNCTION__, msg); return false; } while (0)

static constexpr int kTestBytes = 256 * static_cast<int>(sizeof(int4));
static constexpr int kNumInt4 = kTestBytes / static_cast<int>(sizeof(int4));

__global__ void tc1_tma_load_1d_basic(int4* gmem_in, int4* gmem_out, int num_bytes) {
    extern __shared__ __align__(16) uint8_t smem[];
    auto* smem_int4 = reinterpret_cast<int4*>(smem);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + num_bytes);
    int lane_id = get_lane_id();

    if (lane_id == 0)
        mbarrier_init(mbar, 1);

    tma_load_1d(smem_int4, gmem_in, mbar, num_bytes);
    __syncwarp();

    for (int i = lane_id; i < num_bytes / static_cast<int>(sizeof(int4)); i += 32)
        gmem_out[i] = smem_int4[i];
}

__global__ void tc2_tma_store_1d_basic(int4* gmem_out, int num_bytes) {
    extern __shared__ __align__(16) uint8_t smem[];
    auto* smem_int4 = reinterpret_cast<int4*>(smem);
    int lane_id = get_lane_id();

    for (int i = lane_id; i < num_bytes / static_cast<int>(sizeof(int4)); i += 32) {
        smem_int4[i].x = i;
        smem_int4[i].y = i + 1;
        smem_int4[i].z = i + 2;
        smem_int4[i].w = i + 3;
    }
    __syncwarp();

    tma_store_1d(smem_int4, gmem_out, num_bytes);
}

__global__ void tc3_elect_one_sync_no_deadlock(int4* gmem_in, int4* gmem_out, int num_bytes) {
    extern __shared__ __align__(16) uint8_t smem[];
    auto* smem_int4 = reinterpret_cast<int4*>(smem);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + num_bytes);
    int lane_id = get_lane_id();

    if (elect_one_sync()) {
        mbarrier_init(mbar, 1);
        tma_load_1d(smem_int4, gmem_in, mbar, num_bytes);
    }
    __syncwarp();

    for (int i = lane_id; i < num_bytes / static_cast<int>(sizeof(int4)); i += 32)
        gmem_out[i] = smem_int4[i];
}

__global__ void tc4_full_pipeline(int4* gmem_in, int4* gmem_out, int num_bytes) {
    extern __shared__ __align__(16) uint8_t smem[];
    auto* load_buf = reinterpret_cast<int4*>(smem);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + num_bytes);
    uint32_t phase = 0;

    if (elect_one_sync()) {
        mbarrier_init(mbar, 1);
        fence_barrier_init();
    }
    __syncwarp();

    if (elect_one_sync()) {
        tma_load_1d(load_buf, gmem_in, mbar, num_bytes);
        mbarrier_arrive_and_expect_tx(mbar, num_bytes);
    }
    __syncwarp();

    mbarrier_wait(mbar, phase);

    if (elect_one_sync())
        tma_store_1d(load_buf, gmem_out, num_bytes);
    __syncwarp();

    tma_store_wait<0>();
}

__global__ void tc5_multi_stage_pipeline(int4* gmem_in0, int4* gmem_in1, int4* gmem_out, int num_bytes) {
    extern __shared__ __align__(16) uint8_t smem[];
    constexpr int kNumStages = 2;
    constexpr int kStageSize = kTestBytes + static_cast<int>(sizeof(uint64_t)) * 2;

    int lane_id = get_lane_id();
    uint32_t phase0 = 0;
    uint32_t phase1 = 0;

    if (lane_id < kNumStages) {
        auto mbar_ptr = reinterpret_cast<uint64_t*>(smem + kStageSize * lane_id + kTestBytes);
        mbarrier_init(mbar_ptr, 1);
    }
    __syncwarp();

    auto* buf0 = reinterpret_cast<int4*>(smem);
    auto* mbar0 = reinterpret_cast<uint64_t*>(smem + kTestBytes);
    if (elect_one_sync()) {
        tma_load_1d(buf0, gmem_in0, mbar0, num_bytes);
        mbarrier_arrive_and_expect_tx(mbar0, num_bytes);
    }
    __syncwarp();

    mbarrier_wait(mbar0, phase0);

    auto* buf1 = reinterpret_cast<int4*>(smem + kStageSize);
    auto* mbar1 = reinterpret_cast<uint64_t*>(smem + kStageSize + kTestBytes);
    if (elect_one_sync()) {
        tma_load_1d(buf1, gmem_in1, mbar1, num_bytes);
        mbarrier_arrive_and_expect_tx(mbar1, num_bytes);
    }
    __syncwarp();

    mbarrier_wait(mbar1, phase1);

    for (int i = lane_id; i < num_bytes / static_cast<int>(sizeof(int4)); i += 32) {
        gmem_out[i].x = buf0[i].x + buf1[i].x;
        gmem_out[i].y = buf0[i].y + buf1[i].y;
        gmem_out[i].z = buf0[i].z + buf1[i].z;
        gmem_out[i].w = buf0[i].w + buf1[i].w;
    }
}

__global__ void tc6_boundary_empty(int4* gmem_out) {
    extern __shared__ __align__(16) uint8_t smem[];
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem);
    int lane_id = get_lane_id();

    if (elect_one_sync()) {
        mbarrier_init(mbar, 1);
        tma_load_1d(smem, gmem_out, mbar, 0);
    }
    __syncwarp();
}

__global__ void tc7_concurrent_warps(int4* gmem_in, int4* gmem_out, int num_warps, int num_bytes_per_warp) {
    extern __shared__ __align__(16) uint8_t smem[];
    int warp_id = threadIdx.x / 32;
    int lane_id = get_lane_id();
    int warp_smem_offset = warp_id * (num_bytes_per_warp + static_cast<int>(sizeof(uint64_t)) * 2);

    auto* load_buf = reinterpret_cast<int4*>(smem + warp_smem_offset);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + warp_smem_offset + num_bytes_per_warp);

    if (elect_one_sync()) {
        mbarrier_init(mbar, 1);
        tma_load_1d(load_buf, gmem_in + warp_id * (num_bytes_per_warp / static_cast<int>(sizeof(int4))),
                    mbar, num_bytes_per_warp);
    }
    __syncwarp();

    for (int i = lane_id; i < num_bytes_per_warp / static_cast<int>(sizeof(int4)); i += 32) {
        int4 val = load_buf[i];
        gmem_out[warp_id * (num_bytes_per_warp / static_cast<int>(sizeof(int4))) + i].x = val.x + warp_id;
        gmem_out[warp_id * (num_bytes_per_warp / static_cast<int>(sizeof(int4))) + i].y = val.y + warp_id;
        gmem_out[warp_id * (num_bytes_per_warp / static_cast<int>(sizeof(int4))) + i].z = val.z + warp_id;
        gmem_out[warp_id * (num_bytes_per_warp / static_cast<int>(sizeof(int4))) + i].w = val.w + warp_id;
    }
}

__global__ void tc8a_fixed_mbarrier(int* error_count, int num_iterations) {
    extern __shared__ __align__(16) uint8_t smem[];
    int warp_id = threadIdx.x / 32;
    int lane_id = get_lane_id();

    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem);
    int* shared_val = reinterpret_cast<int*>(smem + sizeof(uint64_t));
    uint32_t phase = 0;

    if (warp_id == 0 && lane_id == 0) {
        *shared_val = 0;
        mbarrier_init(mbar, 1);
    }
    __syncthreads();

    for (int iter = 0; iter < num_iterations; iter++) {
        if (warp_id == 0) {
            if (lane_id == 0) *shared_val = iter + 1;
            __syncwarp();
            if (elect_one_sync())
                mbarrier_arrive_and_expect_tx(mbar, 0);
        } else if (warp_id == 1) {
            mbarrier_wait(mbar, phase);
            __syncwarp();
            if (lane_id == 0) {
                int val = *shared_val;
                if (val != iter + 1) {
                    atomicAdd(error_count, 1);
                }
            }
        }
        __syncthreads();
    }
}

__global__ void tc8b_multi_consumer_mbarrier(int* error_count, int num_iterations) {
    extern __shared__ __align__(16) uint8_t smem[];
    int warp_id = threadIdx.x / 32;
    int lane_id = get_lane_id();

    constexpr int kNumConsumers = 3;
    uint64_t* full_mbar = reinterpret_cast<uint64_t*>(smem);
    uint64_t* empty_mbar = reinterpret_cast<uint64_t*>(smem + sizeof(uint64_t));
    int* shared_val = reinterpret_cast<int*>(smem + sizeof(uint64_t) * 2);

    if (warp_id == 0 && lane_id == 0) {
        *shared_val = 0;
        mbarrier_init(full_mbar, 1);
        mbarrier_init(empty_mbar, kNumConsumers);
    }
    __syncthreads();

    uint32_t producer_phase = 1;
    uint32_t consumer_phase = 0;

    for (int iter = 0; iter < num_iterations; iter++) {
        if (warp_id == 0) {
            mbarrier_wait(empty_mbar, producer_phase);
            __syncwarp();
            if (lane_id == 0) *shared_val = iter + 1;
            __syncwarp();
            if (elect_one_sync())
                mbarrier_arrive_and_expect_tx(full_mbar, 0);
        } else if (warp_id >= 1 && warp_id <= kNumConsumers) {
            mbarrier_wait(full_mbar, consumer_phase);
            __syncwarp();
            if (lane_id == 0) {
                int val = *shared_val;
                if (val != iter + 1) {
                    atomicAdd(error_count, 1);
                }
            }
            __syncwarp();
            if (elect_one_sync())
                mbarrier_arrive(empty_mbar);
        }
    }
}

void fill_int4_pattern(int4* data, int num_int4) {
    for (int i = 0; i < num_int4; i++) {
        data[i].x = i;
        data[i].y = i + 1000;
        data[i].z = i + 2000;
        data[i].w = i + 3000;
    }
}

bool check_int4_pattern(int4* data, int num_int4, const char* label) {
    for (int i = 0; i < num_int4; i++) {
        if (data[i].x != i || data[i].y != i + 1000 ||
            data[i].z != i + 2000 || data[i].w != i + 3000) {
            printf("  [FAIL] %s: mismatch at [%d]: got (%d,%d,%d,%d), expected (%d,%d,%d,%d)\n",
                   label, i, data[i].x, data[i].y, data[i].z, data[i].w,
                   i, i + 1000, i + 2000, i + 3000);
            return false;
        }
    }
    return true;
}

bool test_tc1_load_basic() {
    printf("TC-1: tma_load_1d basic correctness\n");
    int4 *d_in, *d_out;
    int4 h_in[kNumInt4], h_out[kNumInt4];
    fill_int4_pattern(h_in, kNumInt4);

    CUDA_CALL(cudaMalloc(&d_in, kTestBytes));
    CUDA_CALL(cudaMalloc(&d_out, kTestBytes));
    CUDA_CALL(cudaMemcpy(d_in, h_in, kTestBytes, cudaMemcpyHostToDevice));

    int smem = kTestBytes + static_cast<int>(sizeof(uint64_t));
    tc1_tma_load_1d_basic<<<1, 32, smem>>>(d_in, d_out, kTestBytes);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(h_out, d_out, kTestBytes, cudaMemcpyDeviceToHost));
    bool ok = check_int4_pattern(h_out, kNumInt4, "TC-1");

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));

    if (ok) { TEST_PASS(); return true; }
    return false;
}

bool test_tc2_store_basic() {
    printf("TC-2: tma_store_1d basic correctness\n");
    int4 *d_out, h_out[kNumInt4];
    memset(h_out, 0, sizeof(h_out));

    CUDA_CALL(cudaMalloc(&d_out, kTestBytes));
    CUDA_CALL(cudaMemset(d_out, 0, kTestBytes));

    int smem = kTestBytes;
    tc2_tma_store_1d_basic<<<1, 32, smem>>>(d_out, kTestBytes);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(h_out, d_out, kTestBytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < kNumInt4 && ok; i++) {
        if (h_out[i].x != i || h_out[i].y != i + 1 ||
            h_out[i].z != i + 2 || h_out[i].w != i + 3) {
            printf("  [FAIL] TC-2: mismatch at [%d]\n", i);
            ok = false;
        }
    }

    CUDA_CALL(cudaFree(d_out));

    if (ok) { TEST_PASS(); return true; }
    return false;
}

bool test_tc3_elect_one_sync() {
    printf("TC-3: elect_one_sync() no deadlock\n");

    int4 *d_in, *d_out;
    int4 h_in[kNumInt4], h_out[kNumInt4];
    fill_int4_pattern(h_in, kNumInt4);

    CUDA_CALL(cudaMalloc(&d_in, kTestBytes));
    CUDA_CALL(cudaMalloc(&d_out, kTestBytes));
    CUDA_CALL(cudaMemcpy(d_in, h_in, kTestBytes, cudaMemcpyHostToDevice));

    int smem = kTestBytes + static_cast<int>(sizeof(uint64_t));
    tc3_elect_one_sync_no_deadlock<<<1, 32, smem>>>(d_in, d_out, kTestBytes);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(h_out, d_out, kTestBytes, cudaMemcpyDeviceToHost));
    bool ok = check_int4_pattern(h_out, kNumInt4, "TC-3");

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));

    if (ok) { TEST_PASS(); return true; }
    return false;
}

bool test_tc4_full_pipeline() {
    printf("TC-4: Full TMA pipeline (load → mbarrier → wait → store)\n");

    int4 *d_in, *d_out;
    int4 h_in[kNumInt4], h_out[kNumInt4];
    fill_int4_pattern(h_in, kNumInt4);
    memset(h_out, 0, sizeof(h_out));

    CUDA_CALL(cudaMalloc(&d_in, kTestBytes));
    CUDA_CALL(cudaMalloc(&d_out, kTestBytes));
    CUDA_CALL(cudaMemcpy(d_in, h_in, kTestBytes, cudaMemcpyHostToDevice));

    int smem = kTestBytes + static_cast<int>(sizeof(uint64_t));
    tc4_full_pipeline<<<1, 32, smem>>>(d_in, d_out, kTestBytes);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(h_out, d_out, kTestBytes, cudaMemcpyDeviceToHost));
    bool ok = check_int4_pattern(h_out, kNumInt4, "TC-4");

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));

    if (ok) { TEST_PASS(); return true; }
    return false;
}

bool test_tc5_multi_stage() {
    printf("TC-5: Multi-stage pipeline (kNumStages=2)\n");

    int4 *d_in0, *d_in1, *d_out;
    int4 h_in0[kNumInt4], h_in1[kNumInt4], h_out[kNumInt4];
    fill_int4_pattern(h_in0, kNumInt4);
    for (int i = 0; i < kNumInt4; i++) {
        h_in1[i].x = i * 2;
        h_in1[i].y = i * 2 + 1;
        h_in1[i].z = i * 2 + 2;
        h_in1[i].w = i * 2 + 3;
    }

    CUDA_CALL(cudaMalloc(&d_in0, kTestBytes));
    CUDA_CALL(cudaMalloc(&d_in1, kTestBytes));
    CUDA_CALL(cudaMalloc(&d_out, kTestBytes));
    CUDA_CALL(cudaMemcpy(d_in0, h_in0, kTestBytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_in1, h_in1, kTestBytes, cudaMemcpyHostToDevice));

    constexpr int kStageSize = kTestBytes + static_cast<int>(sizeof(uint64_t)) * 2;
    int smem = 2 * kStageSize;
    tc5_multi_stage_pipeline<<<1, 32, smem>>>(d_in0, d_in1, d_out, kTestBytes);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(h_out, d_out, kTestBytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < kNumInt4 && ok; i++) {
        int4 expected;
        expected.x = h_in0[i].x + h_in1[i].x;
        expected.y = h_in0[i].y + h_in1[i].y;
        expected.z = h_in0[i].z + h_in1[i].z;
        expected.w = h_in0[i].w + h_in1[i].w;
        if (h_out[i].x != expected.x || h_out[i].y != expected.y ||
            h_out[i].z != expected.z || h_out[i].w != expected.w) {
            printf("  [FAIL] TC-5: mismatch at [%d]\n", i);
            ok = false;
        }
    }

    CUDA_CALL(cudaFree(d_in0));
    CUDA_CALL(cudaFree(d_in1));
    CUDA_CALL(cudaFree(d_out));

    if (ok) { TEST_PASS(); return true; }
    return false;
}

bool test_tc6_boundary() {
    printf("TC-6: Boundary conditions (num_bytes=0)\n");

    int4 *d_out;
    CUDA_CALL(cudaMalloc(&d_out, sizeof(int4)));

    tc6_boundary_empty<<<1, 32, sizeof(uint64_t)>>>(d_out);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaFree(d_out));

    TEST_PASS();
    return true;
}

bool test_tc7_concurrent_warps() {
    printf("TC-7: Concurrent warps (4 warps, independent buffers)\n");

    constexpr int kNumWarps = 4;
    constexpr int kBytesPerWarp = 64 * static_cast<int>(sizeof(int4));
    constexpr int kNumInt4PerWarp = kBytesPerWarp / static_cast<int>(sizeof(int4));
    constexpr int kTotalBytes = kNumWarps * kBytesPerWarp;
    constexpr int kTotalInt4 = kNumWarps * kNumInt4PerWarp;

    int4 *d_in, *d_out;
    int4 h_in[kTotalInt4], h_out[kTotalInt4];
    fill_int4_pattern(h_in, kTotalInt4);

    CUDA_CALL(cudaMalloc(&d_in, kTotalBytes));
    CUDA_CALL(cudaMalloc(&d_out, kTotalBytes));
    CUDA_CALL(cudaMemcpy(d_in, h_in, kTotalBytes, cudaMemcpyHostToDevice));

    int smem = kNumWarps * (kBytesPerWarp + static_cast<int>(sizeof(uint64_t)) * 2);
    tc7_concurrent_warps<<<1, kNumWarps * 32, smem>>>(d_in, d_out, kNumWarps, kBytesPerWarp);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(h_out, d_out, kTotalBytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int w = 0; w < kNumWarps && ok; w++) {
        for (int i = 0; i < kNumInt4PerWarp && ok; i++) {
            int idx = w * kNumInt4PerWarp + i;
            int4 got = h_out[idx];
            if (got.x != h_in[idx].x + w || got.y != h_in[idx].y + w ||
                got.z != h_in[idx].z + w || got.w != h_in[idx].w + w) {
                printf("  [FAIL] TC-7: warp=%d idx=%d\n", w, idx);
                ok = false;
            }
        }
    }

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));

    if (ok) { TEST_PASS(); return true; }
    return false;
}

bool test_tc8_cross_warp_mbarrier() {
    printf("TC-8: Cross-warp mbarrier synchronization\n");

    constexpr int kNumIterations = 10000;
    int *d_error_a, *d_error_b, h_error_a, h_error_b;

    CUDA_CALL(cudaMalloc(&d_error_a, sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_error_b, sizeof(int)));

    int smem_a = sizeof(int) + sizeof(uint64_t);

    printf("  Running kernel A (fixed mbarrier, elect_one arrive)...\n");
    CUDA_CALL(cudaMemset(d_error_a, 0, sizeof(int)));
    tc8a_fixed_mbarrier<<<1, 64, smem_a>>>(d_error_a, kNumIterations);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(&h_error_a, d_error_a, sizeof(int), cudaMemcpyDeviceToHost));

    printf("  Running kernel B (multi-consumer mbarrier, 3 consumers)...\n");
    CUDA_CALL(cudaMemset(d_error_b, 0, sizeof(int)));
    int smem_b = sizeof(int) + sizeof(uint64_t) * 2;
    tc8b_multi_consumer_mbarrier<<<1, 128, smem_b>>>(d_error_b, kNumIterations);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(&h_error_b, d_error_b, sizeof(int), cudaMemcpyDeviceToHost));

    printf("  Kernel A (1-on-1) errors: %d / %d\n", h_error_a, kNumIterations);
    printf("  Kernel B (1-on-3) errors: %d / %d\n", h_error_b, kNumIterations);

    bool ok = true;
    if (h_error_a > 0) {
        printf("  [FAIL] TC-8: mbarrier 1-on-1 cross-warp sync broken\n");
        ok = false;
    }
    if (h_error_b > 0) {
        printf("  [FAIL] TC-8: mbarrier 1-on-3 cross-warp sync broken\n");
        ok = false;
    }

    CUDA_CALL(cudaFree(d_error_a));
    CUDA_CALL(cudaFree(d_error_b));

    if (ok) { TEST_PASS(); return true; }
    return false;
}

int main() {
    int device = 0;
    CUDA_CALL(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, device));
    printf("Running on: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    int passed = 0, total = 8;
    bool results[8];

    results[0] = test_tc1_load_basic();
    results[1] = test_tc2_store_basic();
    results[2] = test_tc3_elect_one_sync();
    results[3] = test_tc4_full_pipeline();
    results[4] = test_tc5_multi_stage();
    results[5] = test_tc6_boundary();
    results[6] = test_tc7_concurrent_warps();
    results[7] = test_tc8_cross_warp_mbarrier();

    for (int i = 0; i < total; i++)
        if (results[i]) passed++;

    printf("\n=== Results: %d/%d passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}