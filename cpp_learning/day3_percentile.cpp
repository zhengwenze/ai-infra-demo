#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

// ============================================================
// 计算百分位数
// 输入必须已经从小到大排序
// ============================================================

double percentile(const std::vector<double> &sorted_data, double p) {
  if (sorted_data.empty()) {
    return 0.0;
  }

  std::size_t rank =
      static_cast<std::size_t>(std::ceil(p * sorted_data.size()));

  std::size_t index = rank - 1;

  return sorted_data[index];
}

int main() {

  // ========================================================
  // 1. 模拟请求队列
  // ========================================================

  std::queue<int> request_queue;

  request_queue.push(101);
  request_queue.push(102);
  request_queue.push(103);
  request_queue.push(104);
  request_queue.push(105);

  std::cout << "Waiting requests: " << request_queue.size() << std::endl;

  // ========================================================
  // 2. Request ID -> latency
  // ========================================================

  std::unordered_map<int, double> request_latency;

  request_latency[101] = 120;
  request_latency[102] = 80;
  request_latency[103] = 200;
  request_latency[104] = 150;
  request_latency[105] = 100;

  // ========================================================
  // 3. 保存所有 latency
  // ========================================================

  std::vector<double> latencies;

  while (!request_queue.empty()) {

    int request_id = request_queue.front();

    request_queue.pop();

    double latency = request_latency[request_id];

    latencies.push_back(latency);

    std::cout << "Request " << request_id << " latency = " << latency << " ms"
              << std::endl;
  }

  // ========================================================
  // 4. 平均值 average
  // ========================================================

  double sum = 0.0;

  for (double latency : latencies) {
    sum += latency;
  }

  double average = sum / latencies.size();

  // ========================================================
  // 5. 排序
  // ========================================================

  std::sort(latencies.begin(), latencies.end());

  // ========================================================
  // 6. min / max
  // ========================================================

  double min_latency = latencies.front();

  double max_latency = latencies.back();

  // ========================================================
  // 7. percentile
  // ========================================================

  double p50 = percentile(latencies, 0.50);

  double p95 = percentile(latencies, 0.95);

  // ========================================================
  // 8. 输出
  // ========================================================

  std::cout << "\n===== Metrics =====\n";

  std::cout << "Count: " << latencies.size() << std::endl;

  std::cout << "Average: " << average << " ms" << std::endl;

  std::cout << "Min: " << min_latency << " ms" << std::endl;

  std::cout << "Max: " << max_latency << " ms" << std::endl;

  std::cout << "P50: " << p50 << " ms" << std::endl;

  std::cout << "P95: " << p95 << " ms" << std::endl;

  // ========================================================
  // 9. lambda：从大到小重新排序
  // ========================================================

  std::sort(latencies.begin(), latencies.end(),
            [](double a, double b) { return a > b; });

  std::cout << "\nDescending:\n";

  for (double latency : latencies) {

    std::cout << latency << " ms" << std::endl;
  }

  return 0;
}