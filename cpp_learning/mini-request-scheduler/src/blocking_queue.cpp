#include "../include/blocking_queue.h"

#include <stdexcept>

namespace scheduler {

BlockingQueue::BlockingQueue(std::size_t capacity)
    : capacity_(capacity) {
}

void BlockingQueue::push(const Request& request) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (closed_) {
        throw std::runtime_error("BlockingQueue: push on closed queue");
    }

    // 若设了上限且已满，等待 not_full_
    if (capacity_ > 0) {
        not_full_.wait(lock, [this]() {
            return closed_ || queue_.size() < capacity_;
        });
        if (closed_ && queue_.size() >= capacity_) {
            throw std::runtime_error("BlockingQueue: closed while pushing");
        }
    }

    queue_.push_back(request);
    lock.unlock();
    not_empty_.notify_one();
}

void BlockingQueue::push(Request&& request) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (closed_) {
        throw std::runtime_error("BlockingQueue: push on closed queue");
    }

    if (capacity_ > 0) {
        not_full_.wait(lock, [this]() {
            return closed_ || queue_.size() < capacity_;
        });
        if (closed_ && queue_.size() >= capacity_) {
            throw std::runtime_error("BlockingQueue: closed while pushing");
        }
    }

    queue_.push_back(std::move(request));
    lock.unlock();
    not_empty_.notify_one();
}

bool BlockingQueue::pop(Request& out_request) {
    std::unique_lock<std::mutex> lock(mutex_);

    // 队列空且未关闭 → 等待
    not_empty_.wait(lock, [this]() {
        return !queue_.empty() || closed_;
    });

    // 队列空 + 已关闭 → 结束
    if (queue_.empty() && closed_) {
        return false;
    }

    out_request = std::move(queue_.front());
    queue_.pop_front();

    bool has_bounded_capacity = capacity_ > 0;
    lock.unlock();

    // 每取走一个请求都会释放一个槽位，因此每次都要唤醒一个可能正在
    // 等待的生产者。若只在“原队列恰好为满”时通知，一次批量 pop
    // 释放多个槽位后，其余生产者可能永远收不到通知。
    if (has_bounded_capacity) {
        not_full_.notify_one();
    }
    return true;
}

std::size_t BlockingQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

bool BlockingQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

void BlockingQueue::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
}

bool BlockingQueue::isClosed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
}

}  // namespace scheduler
