#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

std::mutex cout_mutex;

void log(const std::string &message) {
  std::lock_guard<std::mutex> lock(cout_mutex);
  std::cout << message << std::endl;
}

// ============================================================
// Request
// 模拟推理服务中的一个请求
// ============================================================

struct Request {
  int id;
  int producer_id;
};

// ============================================================
// BoundedBlockingQueue
// 有界阻塞队列
// ============================================================

template <typename T> class BoundedBlockingQueue {
public:
  explicit BoundedBlockingQueue(std::size_t capacity) : capacity_(capacity) {}

  // --------------------------------------------------------
  // push
  // 成功插入返回 true
  // 队列已经 close 时返回 false
  // --------------------------------------------------------

  bool push(T item) {

    std::unique_lock<std::mutex> lock(mutex_);

    // 队列满了：
    // Producer 在这里睡眠。
    //
    // 只有两种情况允许继续：
    //
    // 1. 队列有空位
    // 2. 队列被关闭
    not_full_.wait(lock,
                   [&]() { return queue_.size() < capacity_ || closed_; });

    // 如果已经关闭，不能再插入新任务
    if (closed_) {
      return false;
    }

    queue_.push(std::move(item));

    // 修改共享数据完成，可以提前释放锁
    lock.unlock();

    // 队列现在肯定不是空的了
    // 唤醒一个等待 Request 的 Consumer
    not_empty_.notify_one();

    return true;
  }

  // --------------------------------------------------------
  // pop
  //
  // 成功取到任务返回 true
  //
  // 如果：
  //   queue 已关闭
  //   &&
  //   queue 已经为空
  //
  // 返回 false，通知 Consumer 安全退出
  // --------------------------------------------------------

  bool pop(T &item) {

    std::unique_lock<std::mutex> lock(mutex_);

    // 队列为空：
    // Consumer 在这里睡眠。
    //
    // 两种情况可以继续：
    //
    // 1. 队列有数据
    // 2. 队列被关闭
    not_empty_.wait(lock, [&]() { return !queue_.empty() || closed_; });

    // closed 并不意味着立即退出。
    //
    // 如果还有残留任务：
    // Consumer 应该继续消费。
    //
    // 只有：
    //
    // closed && empty
    //
    // 才真正退出。
    if (queue_.empty() && closed_) {
      return false;
    }

    item = std::move(queue_.front());
    queue_.pop();

    lock.unlock();

    // pop 之后队列出现了空位，
    // 唤醒一个等待空间的 Producer。
    not_full_.notify_one();

    return true;
  }

  // --------------------------------------------------------
  // close
  //
  // 表示：
  //
  // 不会再接受新的 Request。
  // --------------------------------------------------------

  void close() {

    {
      std::lock_guard<std::mutex> lock(mutex_);

      closed_ = true;
    }

    // 必须唤醒所有等待线程。
    //
    // 否则某些 Producer / Consumer
    // 可能永远睡在那里。
    not_empty_.notify_all();
    not_full_.notify_all();
  }

private:
  std::queue<T> queue_;

  const std::size_t capacity_;

  bool closed_ = false;

  std::mutex mutex_;

  std::condition_variable not_empty_;
  std::condition_variable not_full_;
};

// ============================================================
// Main
// ============================================================

int main() {

  constexpr std::size_t QUEUE_CAPACITY = 3;

  constexpr int PRODUCER_COUNT = 2;
  constexpr int CONSUMER_COUNT = 2;

  constexpr int REQUESTS_PER_PRODUCER = 10;

  BoundedBlockingQueue<Request> request_queue(QUEUE_CAPACITY);

  std::vector<std::thread> producers;
  std::vector<std::thread> consumers;

  log("========================================");
  log("Day 6 - Condition Variable");
  log("Bounded Producer-Consumer Queue");
  log("========================================");

  // ========================================================
  // Consumers
  // ========================================================

  for (int consumer_id = 0; consumer_id < CONSUMER_COUNT; ++consumer_id) {

    consumers.emplace_back([consumer_id, &request_queue]() {
      while (true) {

        Request request;

        auto wait_begin = std::chrono::steady_clock::now();

        bool success = request_queue.pop(request);

        auto wait_end = std::chrono::steady_clock::now();

        auto wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           wait_end - wait_begin)
                           .count();

        if (!success) {

          log("[Consumer " + std::to_string(consumer_id) +
              "] queue closed and empty -> exit");

          break;
        }

        log("[Consumer " + std::to_string(consumer_id) + "] received request " +
            std::to_string(request.id) + " | waited " +
            std::to_string(wait_ms) + " ms");

        // 模拟推理过程
        //
        // Consumer 故意比 Producer 慴，
        // 方便观察：
        //
        // queue 满
        // Producer 阻塞
        std::this_thread::sleep_for(std::chrono::milliseconds(350));

        log("[Consumer " + std::to_string(consumer_id) + "] finished request " +
            std::to_string(request.id));
      }
    });
  }

  // ========================================================
  // Producers
  // ========================================================

  for (int producer_id = 0; producer_id < PRODUCER_COUNT; ++producer_id) {

    producers.emplace_back([producer_id, &request_queue]() {
      for (int i = 0; i < REQUESTS_PER_PRODUCER; ++i) {

        Request request{producer_id * 100 + i, producer_id};

        auto wait_begin = std::chrono::steady_clock::now();

        bool success = request_queue.push(request);

        auto wait_end = std::chrono::steady_clock::now();

        auto wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           wait_end - wait_begin)
                           .count();

        if (!success) {

          log("[Producer " + std::to_string(producer_id) +
              "] queue closed -> stop producing");

          break;
        }

        log("[Producer " + std::to_string(producer_id) +
            "] submitted request " + std::to_string(request.id) +
            " | push waited " + std::to_string(wait_ms) + " ms");

        // Producer 故意比较快
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
      }

      log("[Producer " + std::to_string(producer_id) + "] finished");
    });
  }

  // ========================================================
  // 等待所有 Producer
  // ========================================================

  for (auto &producer : producers) {
    producer.join();
  }

  log("----------------------------------------");
  log("All producers finished.");
  log("Closing request queue...");
  log("----------------------------------------");

  // 不再有新请求
  request_queue.close();

  // ========================================================
  // 等待 Consumer 把剩余请求处理完
  // ========================================================

  for (auto &consumer : consumers) {
    consumer.join();
  }

  log("----------------------------------------");
  log("All consumers exited safely.");
  log("Day 6 experiment finished.");
  log("----------------------------------------");

  return 0;
}