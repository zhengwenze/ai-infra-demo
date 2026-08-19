#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

// ============================================================
// CPU 密集型任务
//
// 每个 i 都进行多轮整数计算。
// result 最终会参与输出，因此编译器不能简单删掉整个计算。
// ============================================================

std::uint64_t cpuWork(std::uint64_t begin, std::uint64_t end) {
  std::uint64_t result = 0;

  for (std::uint64_t i = begin; i < end; ++i) {

    std::uint64_t x = i + 0x9e3779b97f4a7c15ULL;

    // 增加计算量
    for (int j = 0; j < 32; ++j) {

      x ^= x >> 12;
      x ^= x << 25;
      x ^= x >> 27;

      x *= 2685821657736338717ULL;
    }

    result ^= x;
  }

  return result;
}

// ============================================================
// 使用指定数量的线程完成固定总工作量
//
// 关键：
// 不管 thread_count 是多少，TOTAL_WORK 都不变。
// 这样测试才公平。
// ============================================================

double runBenchmark(int thread_count, std::uint64_t total_work,
                    std::uint64_t &final_result) {
  std::vector<std::thread> workers;

  std::vector<std::uint64_t> results(thread_count, 0);

  // 每个线程大约负责多少任务
  std::uint64_t chunk = total_work / thread_count;

  auto start = Clock::now();

  // ========================================================
  // 创建多个 Worker
  // ========================================================

  for (int i = 0; i < thread_count; ++i) {

    std::uint64_t begin = i * chunk;

    // 最后一个线程负责剩余任务
    std::uint64_t end;

    if (i == thread_count - 1) {
      end = total_work;
    } else {
      end = begin + chunk;
    }

    workers.emplace_back(
        [begin, end, i, &results]() { results[i] = cpuWork(begin, end); });
  }

  // ========================================================
  // 等待所有线程完成
  // ========================================================

  for (std::thread &worker : workers) {

    worker.join();
  }

  auto end = Clock::now();

  // ========================================================
  // 合并计算结果
  // ========================================================

  final_result = 0;

  for (std::uint64_t value : results) {

    final_result ^= value;
  }

  // 使用 double 毫秒
  // 不再把 1.8ms 粗暴截断成 1ms
  double duration_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  return duration_ms;
}

// ============================================================
// 对同一种线程数量重复测试
//
// 返回中位数 median。
// ============================================================

double benchmarkMultipleTimes(int thread_count, std::uint64_t total_work,
                              int repeat_count) {
  std::vector<double> times;

  std::cout << "\nTesting " << thread_count << " thread(s)" << std::endl;

  for (int i = 0; i < repeat_count; ++i) {

    std::uint64_t result = 0;

    double time_ms = runBenchmark(thread_count, total_work, result);

    times.push_back(time_ms);

    std::cout << "  Run " << i + 1 << ": " << std::fixed << std::setprecision(3)
              << time_ms << " ms"
              << "    result=" << result << std::endl;
  }

  // ========================================================
  // Day 3 的知识：sort
  // ========================================================

  std::sort(times.begin(), times.end());

  // 取中位数
  double median = times[times.size() / 2];

  return median;
}

// ============================================================
// main
// ============================================================

int main() {

  std::cout << "========================================\n";

  std::cout << "Day 4 - CPU Thread Scaling Benchmark\n";

  std::cout << "========================================\n";

  // ========================================================
  // 检测 CPU 硬件并发能力
  // ========================================================

  unsigned int hardware_threads = std::thread::hardware_concurrency();

  std::cout << "hardware_concurrency = " << hardware_threads << "\n";

  // ========================================================
  // 固定总工作量
  //
  // 如果运行仍然小于 300ms，
  // 可以继续提高到：
  //
  // 100'000'000
  // 200'000'000
  // ========================================================

  const std::uint64_t TOTAL_WORK = 50'000'000;

  // 每个线程数量测试 5 次
  const int REPEAT_COUNT = 5;

  // ========================================================
  // 根据你的 10 hardware concurrency
  // 专门设计测试点
  // ========================================================

  std::vector<int> thread_counts = {

      1,  2,  4, 8,

      10, // 你的硬件并发值

      12, 16, 20};

  // 保存每种线程数量的 median

  std::vector<double> median_times;

  // ========================================================
  // 开始 Benchmark
  // ========================================================

  for (int thread_count : thread_counts) {

    double median =
        benchmarkMultipleTimes(thread_count, TOTAL_WORK, REPEAT_COUNT);

    median_times.push_back(median);

    // 每组之间稍微暂停
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // ========================================================
  // 单线程作为 baseline
  // ========================================================

  double baseline = median_times[0];

  // ========================================================
  // 输出最终结果
  // ========================================================

  std::cout << "\n\n";

  std::cout << "==============================================\n";

  std::cout << "Final Benchmark Result\n";

  std::cout << "==============================================\n";

  std::cout << std::left << std::setw(12) << "Threads"

            << std::setw(18) << "Median(ms)"

            << std::setw(15) << "Speedup"

            << std::setw(15) << "Efficiency"

            << "\n";

  std::cout << "----------------------------------------------"
            << "\n";

  for (std::size_t i = 0; i < thread_counts.size(); ++i) {

    int threads = thread_counts[i];

    double time_ms = median_times[i];

    double speedup = baseline / time_ms;

    // ====================================================
    // 并行效率
    //
    // speedup / threads
    //
    // 例如：
    //
    // 2 threads
    // speedup = 1.8
    //
    // efficiency = 1.8 / 2 = 90%
    // ====================================================

    double efficiency = speedup / threads * 100.0;

    std::cout << std::left << std::setw(12) << threads

              << std::setw(18) << std::fixed << std::setprecision(3) << time_ms

              << std::setw(15) << std::setprecision(2) << speedup

              << std::setw(14) << std::setprecision(1) << efficiency

              << "%"
              << "\n";
  }

  std::cout << "==============================================\n";

  return 0;
}