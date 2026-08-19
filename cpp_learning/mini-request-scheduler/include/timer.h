#ifndef MINI_SCHEDULER_TIMER_H
#define MINI_SCHEDULER_TIMER_H

#include <chrono>
#include <vector>

namespace scheduler {

// ============================================================
// Timer：全局时间锚点工具
// - epoch()：返回程序启动到现在的毫秒数（double）
// - 用 steady_clock，单调递增不受系统时间调整影响
// ============================================================

class Timer {
public:
  // 重置 epoch（main 一开始调用一次即可）
  static void resetEpoch();

  // 返回从 epoch 开始到现在的毫秒数
  static double nowMs();

  // 睡眠 ms
  static void sleepMs(int ms);
};
namespace stats {

// 百分位数（输入必须已排序，p ∈ [0,1]）
double percentile(const std::vector<double> &sorted_data, double p);

// 平均值
double average(const std::vector<double> &data);

} // namespace stats

} // namespace scheduler

#endif // MINI_SCHEDULER_TIMER_H
