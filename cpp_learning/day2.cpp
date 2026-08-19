#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

struct Request {

  int id;

  std::string prompt;

  int max_tokens;
};

// ============================================================
// RequestTimer
// 使用 RAII 自动记录请求执行时间
// ============================================================

class RequestTimer {

private:
  int request_id;

  std::chrono::steady_clock::time_point start_time;

public:
  // ========================================================
  // 构造函数
  // 创建 RequestTimer 时自动执行
  // ========================================================

  RequestTimer(int id)
      : request_id(id), start_time(std::chrono::steady_clock::now()) {

    std::cout << "[Timer] Request " << request_id << " started" << std::endl;
  }

  // ========================================================
  // 析构函数
  // RequestTimer 生命周期结束时自动执行
  // ========================================================

  ~RequestTimer() {

    auto end_time = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    std::cout << "[Timer] Request " << request_id << " took "
              << duration.count() << " ms" << std::endl;
  }
};

// ============================================================
// 模拟模型推理
// ============================================================

void inference(const Request &request) {

  // 创建 timer
  // 构造函数自动开始计时

  RequestTimer timer(request.id);

  std::cout << "[Model] Processing: " << request.prompt << std::endl;

  // 模拟模型计算 500ms

  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  std::cout << "[Model] Finished" << std::endl;

  // inference 函数结束
  //
  // timer 生命周期结束
  //
  // 自动调用 ~RequestTimer()
  //
  // 自动计算耗时
}

// ============================================================
// main
// ============================================================

int main() {

  Request request{1, "What is KV Cache?", 128};

  inference(request);

  return 0;
}