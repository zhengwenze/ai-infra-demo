#ifndef MINI_SCHEDULER_BLOCKING_QUEUE_H
#define MINI_SCHEDULER_BLOCKING_QUEUE_H

#include "request.h"
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>

namespace scheduler {

// ============================================================
// BlockingQueue：线程安全的阻塞队列
// - push()：阻塞插入，若队列满则等待
// - pop()：阻塞取出，若队列空且未关闭则等待
// - close()：关闭队列，所有等待中的 pop 返回 false
// ============================================================

class BlockingQueue {
public:
  // 默认容量上限（0 表示无上限）
  explicit BlockingQueue(std::size_t capacity = 0);
  // 入队：队列满时阻塞直到有空位；若已关闭则抛异常
  void push(const Request &request);
  void push(Request &&request);
  // 出队：队列空时阻塞；若队列空且已关闭则返回 false
  bool pop(Request &out_request);
  // 当前大小
  std::size_t size() const;
  // 是否空
  bool empty() const;
  // 关闭队列：通知所有等待的 pop 结束
  void close();
  // 是否已关闭
  bool isClosed() const;

private:
  mutable std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::deque<Request> queue_;
  std::size_t capacity_;
  bool closed_ = false;
};

} // namespace scheduler

#endif // MINI_SCHEDULER_BLOCKING_QUEUE_H
