#include <algorithm>
#include <atomic>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "../include/blocking_queue.h"
#include "../include/request.h"
#include "../include/timer.h"

using namespace scheduler;

// ============================================================
// 实验参数
// ============================================================

constexpr int kTotalRequests   = 1000;   // 总请求数
constexpr int kProducerThreads = 2;      // 生产者线程数（推送请求）
constexpr int kConsumerThreads = 2;      // 消费者线程数（执行推理）
constexpr int kBatchSize       = 4;      // 批大小
constexpr int kQueueCapacity   = 64;     // 阻塞队列容量（0 = 不限）

// 全局 ID 生成器（原子，防止 Race Condition）
static std::atomic<int> g_next_id{1};

// 全局结果收集（完成的请求，带延迟统计）
static std::vector<Request> g_completed;
static std::mutex g_completed_mutex;

// ============================================================
// FakeModel：模拟 LLM 推理
// 按 max_tokens × 0.3ms 睡眠
// ============================================================

class FakeModel {
public:
    std::string generate(const Request& req) {
        int sleep_ms = static_cast<int>(req.max_tokens * 0.3);
        Timer::sleepMs(sleep_ms);
        return "[FakeModel] response for request " + std::to_string(req.id);
    }
};

// ============================================================
// Producer：产生请求并 push 进 BlockingQueue
// ============================================================

void producerThread(BlockingQueue& queue, int count) {
    std::mt19937 rng(static_cast<uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id())));

    // max_tokens：大部分在 64~256，小部分长尾 512~1024
    std::uniform_int_distribution<int> dist_normal(64, 256);
    std::uniform_int_distribution<int> dist_tail(512, 1024);

    // prompt 池
    const std::vector<std::string> prompts = {
        "What is KV Cache?",
        "What is Continuous Batching?",
        "What is PagedAttention?",
        "Explain vLLM architecture",
        "What is TensorRT-LLM?",
        "How does Flash Attention work?",
        "Compare GPT and BERT",
        "What is speculative decoding?",
    };
    std::uniform_int_distribution<std::size_t> prompt_dist(0, prompts.size() - 1);

    for (int i = 0; i < count; ++i) {
        int id = g_next_id.fetch_add(1);

        int max_tokens = (i % 10 == 0) ? dist_tail(rng) : dist_normal(rng);

        const std::string& prompt = prompts[prompt_dist(rng)];

        Request req(id, prompt, max_tokens);

        // 记录入队时刻（arrive_time）
        req.arrive_time_ms = Timer::nowMs();

        queue.push(req);

        // 产生新请求的间隔：1~5ms，模拟用户流量
        std::uniform_int_distribution<int> gap_dist(1, 5);
        Timer::sleepMs(gap_dist(rng));
    }
}

// ============================================================
// Consumer：从 BlockingQueue 取一批请求，推理，统计耗时
// ============================================================

void consumerThread(BlockingQueue& queue) {
    FakeModel model;

    while (true) {
        std::vector<Request> batch;
        batch.reserve(kBatchSize);

        // 取出最多 kBatchSize 个请求
        for (int i = 0; i < kBatchSize; ++i) {
            Request req;
            bool ok = queue.pop(req);
            if (!ok) break;   // 队列关闭且空 → 结束
            req.start_time_ms = Timer::nowMs();   // 开始推理时刻
            batch.push_back(std::move(req));

            // 若队列暂时空但未关闭，不继续阻塞等更多（凑不满一批也跑）
            if (queue.empty() && !queue.isClosed()) break;
        }

        if (batch.empty()) break;   // 真的结束了

        // 模拟 batch 推理
        for (Request& req : batch) {
            model.generate(req);  // 这里 sleep

            req.finish_time_ms = Timer::nowMs();
            req.latency_ms = req.finish_time_ms - req.arrive_time_ms;

            // 汇总到全局结果表
            {
                std::lock_guard<std::mutex> lock(g_completed_mutex);
                g_completed.push_back(std::move(req));
            }
        }
    }
}

// ============================================================
// 打印统计结果
// ============================================================

void printReport() {
    std::vector<double> latencies;
    std::vector<double> queue_times;   // 排队时间 = start - arrive
    std::vector<double> infer_times;   // 推理时间 = finish - start

    latencies.reserve(g_completed.size());
    queue_times.reserve(g_completed.size());
    infer_times.reserve(g_completed.size());

    for (const Request& r : g_completed) {
        latencies.push_back(r.latency_ms);
        queue_times.push_back(r.start_time_ms - r.arrive_time_ms);
        infer_times.push_back(r.finish_time_ms - r.start_time_ms);
    }

    std::sort(latencies.begin(), latencies.end());
    std::sort(queue_times.begin(), queue_times.end());
    std::sort(infer_times.begin(), infer_times.end());

    std::cout << "\n";
    std::cout << "=============================================================\n";
    std::cout << "                   Mini Request Scheduler Report            \n";
    std::cout << "=============================================================\n";
    std::cout << "  Config:\n";
    std::cout << "    Total Requests    : " << kTotalRequests << "\n";
    std::cout << "    Producer Threads  : " << kProducerThreads << "\n";
    std::cout << "    Consumer Threads  : " << kConsumerThreads << "\n";
    std::cout << "    Batch Size        : " << kBatchSize << "\n";
    std::cout << "    Queue Capacity    : "
              << (kQueueCapacity == 0 ? "unlimited" : std::to_string(kQueueCapacity))
              << "\n";
    std::cout << "    Completed         : " << g_completed.size() << "\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  Metrics (单位: ms)\n";
    std::cout << "-------------------------------------------------------------\n";

    auto print_row = [&](const char* name,
                         const std::vector<double>& sorted) {
        printf("  %-18s  min=%7.2f  P50=%7.2f  P90=%7.2f  "
               "P95=%7.2f  P99=%7.2f  max=%7.2f  avg=%7.2f\n",
               name,
               sorted.front(),
               stats::percentile(sorted, 0.50),
               stats::percentile(sorted, 0.90),
               stats::percentile(sorted, 0.95),
               stats::percentile(sorted, 0.99),
               sorted.back(),
               stats::average(sorted));
    };

    print_row("E2E Latency",    latencies);
    print_row("Queue Wait",     queue_times);
    print_row("Inference Time", infer_times);
    std::cout << "=============================================================\n";
    std::cout << std::endl;
}

// ============================================================
// main
// ============================================================

int main() {
    // 关闭 stdout 缓冲，确保脚本/管道环境也能实时看到输出
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    Timer::resetEpoch();

    std::cout << "=============================================================\n";
    std::cout << " Mini Request Scheduler : "
              << kProducerThreads << " producers + "
              << kConsumerThreads << " consumers\n";
    std::cout << "=============================================================\n\n";

    BlockingQueue queue(kQueueCapacity);

    // ===== 启动消费者（先于生产者，避免队列爆满） =====
    std::vector<std::thread> consumers;
    consumers.reserve(kConsumerThreads);
    for (int i = 0; i < kConsumerThreads; ++i) {
        consumers.emplace_back(consumerThread, std::ref(queue));
    }

    // ===== 启动生产者 =====
    int req_per_producer = kTotalRequests / kProducerThreads;
    int remainder        = kTotalRequests % kProducerThreads;

    std::vector<std::thread> producers;
    producers.reserve(kProducerThreads);
    for (int i = 0; i < kProducerThreads; ++i) {
        int count = req_per_producer + (i == 0 ? remainder : 0);
        producers.emplace_back(producerThread, std::ref(queue), count);
    }

    // ===== 等待所有生产者结束，再关闭队列 =====
    for (auto& t : producers) t.join();
    std::cout << "[Main] All producers finished. Closing queue...\n";
    queue.close();

    // ===== 等待所有消费者结束 =====
    for (auto& t : consumers) t.join();
    std::cout << "[Main] All consumers finished. Requests completed: "
              << g_completed.size() << "\n";

    // ===== 打印报告 =====
    printReport();

    // 完整性检查：丢了请求就报错
    if (static_cast<int>(g_completed.size()) != kTotalRequests) {
        std::cerr << "ERROR: lost "
                  << (kTotalRequests - (int)g_completed.size())
                  << " requests!\n";
        return 1;
    }
    return 0;
}
