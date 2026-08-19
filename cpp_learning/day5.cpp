#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>

// ============================================================
// 实验 1：故意制造 Race Condition
// 两个线程各加 1,000,000 次，期望得到 2,000,000
// counter++ 不是原子操作（读-改-写），会被线程切换打断
// ============================================================

int counter_race = 0;

void increment_race() {
  for (int i = 0; i < 1'000'000; ++i) {
    // 这里不是原子的：
    // 1. 读取 counter_race 的值（register）
    // 2. register + 1
    // 3. 写回 counter_race
    // 如果在 1 和 3 之间被切换，两次自增会"合并"成一次
    counter_race++;
  }
}

void run_experiment1(int times) {
  std::cout << "==================================================\n";
  std::cout << "实验 1：Race Condition（无保护）\n";
  std::cout << "Expected: 2000000\n";
  std::cout << "--------------------------------------------------\n";
  for (int t = 0; t < times; ++t) {
    counter_race = 0;
    std::thread t1(increment_race);
    std::thread t2(increment_race);
    t1.join();
    t2.join();
    std::cout << "Run #" << (t + 1) << "  Actual: " << counter_race;
    if (counter_race != 2'000'000) {
      int loss = 2'000'000 - counter_race;
      std::cout << "   ❌ 丢失了 " << loss << " 次自增 ("
                << (loss * 100.0 / 2'000'000) << "%)";
    } else {
      std::cout << "   ✅ 偶然正确（不代表没有问题）";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

// ============================================================
// 实验 2：用 mutex 修复 Race Condition
// std::mutex 的 lock/unlock 保证同一时刻只有一个线程
// 能进入临界区（critical section）
// ============================================================

int counter_mutex = 0;
std::mutex mtx;

void increment_mutex() {
  for (int i = 0; i < 1'000'000; ++i) {
    std::lock_guard<std::mutex> lock(mtx); // 构造即 lock
    counter_mutex++;                       // 临界区
  } // 析构即 unlock（异常也安全）
}

void run_experiment2(int times) {
  std::cout << "==================================================\n";
  std::cout << "实验 2：用 std::mutex 修复\n";
  std::cout << "Expected: 2000000\n";
  std::cout << "--------------------------------------------------\n";
  for (int t = 0; t < times; ++t) {
    counter_mutex = 0;
    std::thread t1(increment_mutex);
    std::thread t2(increment_mutex);
    t1.join();
    t2.join();
    std::cout << "Run #" << (t + 1) << "  Actual: " << counter_mutex;
    if (counter_mutex == 2'000'000) {
      std::cout << "   ✅ 正确";
    } else {
      std::cout << "   ❌ 错误";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

// ============================================================
// 实验 3：用 std::atomic 修复 Race Condition
// atomic<int> 的 operator++ 是硬件级原子操作
// 比 mutex 更轻量（无锁）
// ============================================================

std::atomic<int> counter_atomic{0};

void increment_atomic() {
  for (int i = 0; i < 1'000'000; ++i) {
    counter_atomic++; // std::atomic 保证原子性
  }
}

void run_experiment3(int times) {
  std::cout << "==================================================\n";
  std::cout << "实验 3：用 std::atomic 修复（无锁）\n";
  std::cout << "Expected: 2000000\n";
  std::cout << "--------------------------------------------------\n";
  for (int t = 0; t < times; ++t) {
    counter_atomic = 0;
    std::thread t1(increment_atomic);
    std::thread t2(increment_atomic);
    t1.join();
    t2.join();
    std::cout << "Run #" << (t + 1) << "  Actual: " << counter_atomic;
    if (counter_atomic == 2'000'000) {
      std::cout << "   ✅ 正确";
    } else {
      std::cout << "   ❌ 错误";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

// ============================================================
// 实验 4：为什么 Race Condition 会发生？
// 拆解 counter++ 的汇编级三步
// 用 yield 故意"放大"这个窗口，更稳定复现
// ============================================================

int counter_explicit = 0;

void increment_explicit() {
  for (int i = 0; i < 100'000; ++i) {
    // 手动拆成三步，中间插入 yield 模拟线程切换
    int tmp = counter_explicit; // 步骤 1：读
    std::this_thread::yield();  // 故意让调度器切换！
    tmp = tmp + 1;              // 步骤 2：改
    counter_explicit = tmp;     // 步骤 3：写
  }
}

void run_experiment4(int times) {
  std::cout << "==================================================\n";
  std::cout << "实验 4：显式拆解 counter++ 三步（放大 Race 窗口）\n";
  std::cout << "Expected: 200000（因为是 10万 * 2 线程）\n";
  std::cout << "--------------------------------------------------\n";
  for (int t = 0; t < times; ++t) {
    counter_explicit = 0;
    std::thread t1(increment_explicit);
    std::thread t2(increment_explicit);
    t1.join();
    t2.join();
    std::cout << "Run #" << (t + 1) << "  Actual: " << counter_explicit;
    int loss = 200'000 - counter_explicit;
    std::cout << "   丢失率 = " << (loss * 100.0 / 200'000) << "%\n";
  }
  std::cout << "\n";
  std::cout << "原理图示：\n";
  std::cout << "  Thread A: 读(10) → yield → 改(11) → 写(11)\n";
  std::cout << "  Thread B:          读(10) → 改(11) → 写(11)\n";
  std::cout << "  结果：两次自增只加了 1，损失 1 次\n\n";
}

// ============================================================
// main
// ============================================================

int main() {
  std::cout << "\n";
  std::cout << "##################################################\n";
  std::cout << "#          Day5：Race Condition 实验             #\n";
  std::cout << "##################################################\n\n";

  run_experiment1(8); // 不加保护，8 次几乎次次错误
  run_experiment2(5); // mutex 保护，5 次全对
  run_experiment3(5); // atomic 保护，5 次全对
  run_experiment4(5); // 放大 Race 窗口，丢失率极高

  std::cout << "==================================================\n";
  std::cout << "总结：\n";
  std::cout << "  1. counter++ 不是原子操作（读-改-写三步）\n";
  std::cout << "  2. 多线程不加保护 → Race Condition → 结果偏小\n";
  std::cout << "  3. 修复方式 1：std::mutex（锁，通用但开销大）\n";
  std::cout << "  4. 修复方式 2：std::atomic（无锁，仅适用于基本类型）\n";
  std::cout << "==================================================\n";

  return 0;
}
