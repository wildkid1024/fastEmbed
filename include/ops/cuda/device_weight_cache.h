#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <unordered_map>

// 设备端权重常驻缓存（消除"每次 GEMM 从主机内存重新上传权重"的问题）
//
// 背景：此前每次 matrix_multiply_transpose / mlp_fused 等算子都会把权重
// cudaMemcpy 到显存。以 Qwen3-Embedding-0.6B 为例，每次前向要向显存搬运
// ~1.76GB 权重（pageable 内存约 6GB/s），折合约 293ms 的固定延迟，与序列
// 长度无关，是 HTTP 模式下短文本 300ms 延迟的主要来源。
//
// 方案：以 host 权重向量的 data() 指针作为 key，首次使用时上传到显存并
// 永久驻留；之后所有前向直接复用设备指针，权重流量从"每请求 GB 级"降为 0。
//
// 使用契约（必须满足，否则会读到过期数据）：
//   1. 传入的 host 向量在模型生命周期内内容不变。当前实现中 weights 表
//      在模型加载完成后不再修改，满足该契约；
//   2. 不要用该缓存承载会被原地修改或复用同一段内存的激活值/临时缓冲。
//
// 线程安全：全局表用一把锁保护；每个权重槽位有独立的互斥锁做双重检查，
// 并发首次上传同一权重时会串行等待，保证只上传一份。
class DeviceWeightCache {
public:
    static DeviceWeightCache& instance();

    // 返回驻留显存的设备指针；首次调用时上传 host_data 的前 count 个 float。
    // host_data 为空或 count 为 0 时返回 nullptr。
    const float* get(const float* host_data, size_t count);

    // 统计信息（用于启动日志与显存占用排查）
    size_t tensor_count() const;
    size_t total_bytes() const;

    // 释放全部驻留显存。仅建议在测试或显式卸载模型时调用（要求当时没有
    // 其它线程正在首次上传）；正常服务进程退出时由 CUDA 上下文统一回收
    //（析构函数有意不释放，避免静态析构顺序问题）。
    void clear();

private:
    DeviceWeightCache() = default;

    // 每个权重一个槽位：槽位锁做双重检查，避免并发首传重复上传
    struct Slot {
        std::mutex mu;
        const float* device_ptr = nullptr;
        size_t bytes = 0;
    };

    mutable std::mutex map_mutex_;
    std::unordered_map<const float*, Slot*> slots_;
    std::atomic<size_t> total_bytes_{0};  // 驻留总字节数（原子维护，避免统计时加槽位锁）
};
