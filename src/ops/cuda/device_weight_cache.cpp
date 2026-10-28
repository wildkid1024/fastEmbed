#include "ops/cuda/device_weight_cache.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("DeviceWeightCache ") + what +
                                 " 失败: " + cudaGetErrorString(err));
    }
}

}  // namespace

DeviceWeightCache& DeviceWeightCache::instance() {
    static DeviceWeightCache cache;
    return cache;
}

const float* DeviceWeightCache::get(const float* host_data, size_t count) {
    if (host_data == nullptr || count == 0) {
        return nullptr;
    }

    // 找到（或创建）该权重对应的槽位，之后在槽位锁内做双重检查
    Slot* slot = nullptr;
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        slot = slots_[host_data];
        if (slot == nullptr) {
            slot = new Slot();
            slots_[host_data] = slot;
        }
    }

    std::lock_guard<std::mutex> slot_lock(slot->mu);
    if (slot->device_ptr != nullptr) {
        return slot->device_ptr;  // 快路径：已驻留
    }

    // 慢路径：首次上传。持槽位锁执行，同一权重的并发首传在此串行等待，
    // 保证只上传一份
    const size_t bytes = count * sizeof(float);
    float* device_ptr = nullptr;
    cuda_check(cudaMalloc(&device_ptr, bytes), "cudaMalloc");
    cuda_check(cudaMemcpy(device_ptr, host_data, bytes, cudaMemcpyHostToDevice),
               "权重上传");

    slot->device_ptr = device_ptr;
    slot->bytes = bytes;
    total_bytes_.fetch_add(bytes, std::memory_order_relaxed);

    // 仅对较大的张量打日志，避免刷屏；累计值便于核对显存占用
    if (bytes >= (4u << 20)) {
        const size_t total = total_bytes_.load(std::memory_order_relaxed);
        size_t count = 0;
        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            count = slots_.size();
        }
        std::printf("[DeviceWeightCache] 驻留权重 %.1f MB（累计 %.2f GB / %zu 个张量）\n",
                    static_cast<double>(bytes) / (1024.0 * 1024.0),
                    static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0),
                    count);
    }
    return device_ptr;
}

size_t DeviceWeightCache::tensor_count() const {
    std::lock_guard<std::mutex> lock(map_mutex_);
    return slots_.size();
}

size_t DeviceWeightCache::total_bytes() const {
    return total_bytes_.load(std::memory_order_relaxed);
}

void DeviceWeightCache::clear() {
    // 注意：调用方需保证此刻没有其它线程正在 get() 首次上传
    std::lock_guard<std::mutex> lock(map_mutex_);
    for (auto& kv : slots_) {
        Slot* slot = kv.second;
        if (slot == nullptr) {
            continue;
        }
        std::lock_guard<std::mutex> slot_lock(slot->mu);
        if (slot->device_ptr != nullptr) {
            cudaFree(const_cast<void*>(static_cast<const void*>(slot->device_ptr)));
            slot->device_ptr = nullptr;
            slot->bytes = 0;
        }
        delete slot;
    }
    slots_.clear();
    total_bytes_.store(0, std::memory_order_relaxed);
}
