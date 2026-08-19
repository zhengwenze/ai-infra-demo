#include "../include/timer.h"

#include <cmath>
#include <numeric>
#include <thread>

namespace scheduler {

// 全局 epoch 起点（静态存储）
// 首次使用时默认初始化；resetEpoch 可重设
static std::chrono::steady_clock::time_point g_epoch =
    std::chrono::steady_clock::now();

void Timer::resetEpoch() {
    g_epoch = std::chrono::steady_clock::now();
}

double Timer::nowMs() {
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = now - g_epoch;
    return duration.count();
}

void Timer::sleepMs(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

namespace stats {

double percentile(const std::vector<double>& sorted_data, double p) {
    if (sorted_data.empty()) {
        return 0.0;
    }
    if (p <= 0.0) return sorted_data.front();
    if (p >= 1.0) return sorted_data.back();

    std::size_t rank = static_cast<std::size_t>(
        std::ceil(p * sorted_data.size()));
    if (rank < 1) rank = 1;
    std::size_t index = rank - 1;
    if (index >= sorted_data.size()) {
        index = sorted_data.size() - 1;
    }
    return sorted_data[index];
}

double average(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

}  // namespace stats

}  // namespace scheduler
