/**
 * Sum the energy buffer.
 */

// Compile this without fast math, and auto-clear the result buffer.
// Source: https://en.wikipedia.org/wiki/Kahan_summation_algorithm

__attribute__((__always_inline__))
void kahanSum(float input, thread float &sum, thread float &c, bool sumLarger) {
    float t = sum + input;
    if (sumLarger) {
        c += (sum - t) + input;
    } else {
        c += (input - t) + sum;
    }
    sum = t;
};

__kernel void reduceEnergy(GLOBAL const float* restrict energyBuffer, GLOBAL atomic_float* restrict result, int bufferSize, int elementsPerWorkGroup, DISPATCH_ARGUMENTS) {
    float sum = 0;
    float c = 0;
    int groupStart = GROUP_ID * elementsPerWorkGroup;
    int groupEnd = min(groupStart + elementsPerWorkGroup, bufferSize);
    for (uint index = groupStart + LOCAL_ID; index < groupEnd; index += GROUP_SIZE) {
        float value = energyBuffer[index];
        kahanSum(value, sum, c, /*sumLarger=*/true);
    }
    
    // Although threadgroup bandwidth might not be a significant bottleneck,
    // threadgroup latency will be. We do SIMD shuffling instead, even though
    // it increases executable size.
    for (uint offset=SIMD_SIZE/2; offset>0; offset/=2) {
        float other_sum = simd_shuffle_down(sum, offset);
        float other_c = simd_shuffle_down(c, offset);
        kahanSum(other_sum, sum, c, abs(sum) >= abs(other_sum))
        kahanSum(other_c, sum, c, /*sumLarger=*/true);
    }
    
    threadgroup float sum_buffer[32 + 1];
    threadgroup float c_buffer[32 + 1];
    if (simd_is_first()) {
        sum_buffer[SIMD_ID] = sum;
        c_buffer[SIMD_ID] = c;
    }
    if (LOCAL_ID >= LOCAL_SIZE/SIMD_SIZE) {
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = sum_buffer[SIMD_LANE_ID];
    c = c_buffer[SIMD_LANE_ID];
    
    for (uint offset=SIMD_SIZE/2; offset>0; offset/=2) {
        float other_sum = simd_shuffle_down(sum, offset);
        float other_c = simd_shuffle_down(c, offset);
        kahanSum(other_sum, sum, c, abs(sum) >= abs(other_sum))
        kahanSum(other_c, sum, c, /*sumLarger=*/true);
    }
    sum += c;
    if (simd_is_first()) {
        atomic_fetch_add_explicit(result, sum, memory_order_relaxed);
    }
}

