#include "blocking_queue.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

using namespace scheduler;
using namespace std::chrono_literals;

int main() {
  BlockingQueue queue(2);
  queue.push(Request(1, "initial-1", 1));
  queue.push(Request(2, "initial-2", 1));

  std::atomic<int> producers_started{0};
  std::atomic<int> producers_completed{0};

  auto producer = [&](int id) {
    producers_started.fetch_add(1);
    queue.push(Request(id, "waiting-producer", 1));
    producers_completed.fetch_add(1);
  };

  std::thread first(producer, 3);
  std::thread second(producer, 4);

  while (producers_started.load() != 2) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(20ms);

  Request request;
  queue.pop(request);
  queue.pop(request);

  const auto deadline = std::chrono::steady_clock::now() + 500ms;
  while (producers_completed.load() != 2 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }

  const bool all_producers_woke = producers_completed.load() == 2;
  queue.close();
  first.join();
  second.join();

  if (!all_producers_woke) {
    std::cerr << "Expected both waiting producers to wake after two pops\n";
    return 1;
  }

  std::cout << "Both waiting producers were notified\n";
  return 0;
}
