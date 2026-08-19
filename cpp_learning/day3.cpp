#include <chrono>
#include <iostream>
#include <thread>

class RequestTimer {
private:
  int request_id;
  std::chrono::steady_clock::time_point start_time;

public:
  // 构造函数：记录请求 id 并启动计时
  RequestTimer(int id)
      : request_id(id), start_time(std::chrono::steady_clock::now()) {}

  // setter：设置 request_id
  void setRequestId(int id) { request_id = id; }

  // 返回耗时（毫秒）
  double elapsed() const {
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    return duration.count();
  }

  void printId() {
    std::cout << "request_id = " << request_id << ", elapsed = " << elapsed()
              << " ms" << std::endl;
  }
};

int main() {
  RequestTimer timer(1);

  // 模拟推理耗时
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  timer.printId();
  return 0;
}