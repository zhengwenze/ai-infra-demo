#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
// 实验参数（默认值，可由命令行覆盖）
// ============================================================

struct Config {
    int total_requests   = 200;     // 总请求数（缩小以加快对照实验）
    int producer_threads = 2;       // 生产者线程数
    int consumer_threads = 2;       // 消费者线程数
    int batch_size       = 4;       // 批大小
    int queue_capacity   = 16;      // 队列容量
    int arrival_gap_ms   = 2;       // 请求到达间隔（生产者间隔）
};

// 全局 ID 生成器
static std::atomic<int> g_next_id{1};

// 全局结果收集
static std::vector<Request> g_completed;
static std::mutex g_completed_mutex;

// 全局 tokens 统计
static std::atomic<long long> g_total_tokens{0};

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

void producerThread(BlockingQueue& queue, int count, int gap_ms) {
    std::mt19937 rng(static_cast<uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id())));

    std::uniform_int_distribution<int> dist_normal(64, 256);
    std::uniform_int_distribution<int> dist_tail(512, 1024);

    const std::vector<std::string> prompts = {
        "What is KV Cache?",
        "What is Continuous Batching?",
        "What is PagedAttention?",
        "Explain vLLM architecture",
        "What is TensorRT-LLM?",
    };
    std::uniform_int_distribution<std::size_t> prompt_dist(0, prompts.size() - 1);

    for (int i = 0; i < count; ++i) {
        int id = g_next_id.fetch_add(1);
        int max_tokens = (i % 10 == 0) ? dist_tail(rng) : dist_normal(rng);
        const std::string& prompt = prompts[prompt_dist(rng)];

        Request req(id, prompt, max_tokens);
        req.arrive_time_ms = Timer::nowMs();

        queue.push(req);

        if (gap_ms > 0) {
            std::uniform_int_distribution<int> gap_dist(
                std::max(1, gap_ms - 1), gap_ms + 1);
            Timer::sleepMs(gap_dist(rng));
        }
    }
}

// ============================================================
// Consumer：从 BlockingQueue 取一批请求，推理，统计耗时
// 修复点：start_time_ms 在真正调用 generate() 前才记录
// ============================================================

void consumerThread(BlockingQueue& queue, int batch_size) {
    FakeModel model;

    while (true) {
        std::vector<Request> batch;
        batch.reserve(batch_size);

        // 取出最多 batch_size 个请求（这一步只算"出队时刻"，不算推理时刻）
        for (int i = 0; i < batch_size; ++i) {
            Request req;
            bool ok = queue.pop(req);
            if (!ok) break;
            batch.push_back(std::move(req));
            if (queue.empty() && !queue.isClosed()) break;
        }

        if (batch.empty()) break;

        // 对每个请求：先记录 start_time，再 generate()，再记录 finish_time
        // 这样 Inference Time 只包含自己的推理时间，不包含同 batch 前面请求的
        for (Request& req : batch) {
            req.start_time_ms = Timer::nowMs();  // ← 修复：真正开始推理前
            model.generate(req);
            req.finish_time_ms = Timer::nowMs();
            req.latency_ms = req.finish_time_ms - req.arrive_time_ms;

            g_total_tokens += req.max_tokens;

            {
                std::lock_guard<std::mutex> lock(g_completed_mutex);
                g_completed.push_back(std::move(req));
            }
        }
    }
}

// ============================================================
// 打印统计结果（含吞吐量）
// ============================================================

void printReport(const Config& cfg, double wall_time_s) {
    std::vector<double> latencies;
    std::vector<double> queue_times;
    std::vector<double> infer_times;

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

    long long total_tokens = g_total_tokens.load();
    double req_per_s = g_completed.size() / wall_time_s;
    double tok_per_s  = total_tokens / wall_time_s;

    std::cout << "\n";
    std::cout << "=============================================================\n";
    std::cout << "                   Mini Request Scheduler Report            \n";
    std::cout << "=============================================================\n";
    std::cout << "  Config:\n";
    std::cout << "    Total Requests   : " << cfg.total_requests << "\n";
    std::cout << "    Producer Threads : " << cfg.producer_threads << "\n";
    std::cout << "    Consumer Threads : " << cfg.consumer_threads << "\n";
    std::cout << "    Batch Size       : " << cfg.batch_size << "\n";
    std::cout << "    Queue Capacity   : " << cfg.queue_capacity << "\n";
    std::cout << "    Arrival Gap      : " << cfg.arrival_gap_ms << " ms\n";
    std::cout << "    Completed        : " << g_completed.size() << "\n";
    std::cout << "    Wall Time        : " << wall_time_s << " s\n";
    std::cout << "    Total Tokens     : " << total_tokens << "\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  Throughput\n";
    std::cout << "    Requests/s       : " << req_per_s << "\n";
    std::cout << "    Tokens/s         : " << tok_per_s << "\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  Latency (ms)\n";
    std::cout << "-------------------------------------------------------------\n";

    auto print_row = [&](const char* name,
                         const std::vector<double>& sorted) {
        if (sorted.empty()) {
            printf("  %-18s  (no data)\n", name);
            return;
        }
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

    // CSV 行：方便脚本聚合多个实验
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  CSV: consumers,batch,queue,req_s,tok_s,"
              << "p50_lat,p90_lat,p99_lat,p50_qw,p99_qw,p50_inf,p99_inf\n";
    printf("  %d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
           cfg.consumer_threads, cfg.batch_size, cfg.queue_capacity,
           req_per_s, tok_per_s,
           stats::percentile(latencies, 0.50),
           stats::percentile(latencies, 0.90),
           stats::percentile(latencies, 0.99),
           stats::percentile(queue_times, 0.50),
           stats::percentile(queue_times, 0.99),
           stats::percentile(infer_times, 0.50),
           stats::percentile(infer_times, 0.99));
    std::cout << "=============================================================\n";
    std::cout << std::endl;
}

// ============================================================
// 解析命令行参数
//   --consumers N --batch N --queue N --requests N --producers N --gap N
// ============================================================

Config parseArgs(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> int {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << "\n";
                std::exit(1);
            }
            return std::atoi(argv[++i]);
        };
        if      (arg == "--consumers") cfg.consumer_threads = next();
        else if (arg == "--batch")     cfg.batch_size = next();
        else if (arg == "--queue")     cfg.queue_capacity = next();
        else if (arg == "--requests")  cfg.total_requests = next();
        else if (arg == "--producers") cfg.producer_threads = next();
        else if (arg == "--gap")       cfg.arrival_gap_ms = next();
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: mini-scheduler [--consumers N] [--batch N] "
                      << "[--queue N] [--requests N] [--producers N] [--gap N]\n";
            std::exit(0);
        }
    }
    return cfg;
}

// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    Config cfg = parseArgs(argc, argv);

    Timer::resetEpoch();

    std::cout << "=============================================================\n";
    std::cout << " Mini Request Scheduler : "
              << cfg.producer_threads << " producers + "
              << cfg.consumer_threads << " consumers"
              << " (batch=" << cfg.batch_size
              << ", queue=" << cfg.queue_capacity << ")\n";
    std::cout << "=============================================================\n\n";

    BlockingQueue queue(cfg.queue_capacity);

    // 启动消费者
    std::vector<std::thread> consumers;
    consumers.reserve(cfg.consumer_threads);
    for (int i = 0; i < cfg.consumer_threads; ++i) {
        consumers.emplace_back(consumerThread,
                               std::ref(queue), cfg.batch_size);
    }

    // 启动生产者
    int req_per_producer = cfg.total_requests / cfg.producer_threads;
    int remainder        = cfg.total_requests % cfg.producer_threads;
    std::vector<std::thread> producers;
    producers.reserve(cfg.producer_threads);
    for (int i = 0; i < cfg.producer_threads; ++i) {
        int count = req_per_producer + (i == 0 ? remainder : 0);
        producers.emplace_back(producerThread,
                               std::ref(queue), count, cfg.arrival_gap_ms);
    }

    // 计时开始（用 wall clock）
    auto wall_start = std::chrono::steady_clock::now();

    // 等待所有生产者结束
    for (auto& t : producers) t.join();
    std::cout << "[Main] All producers finished. Closing queue...\n";
    queue.close();

    // 等待所有消费者结束
    for (auto& t : consumers) t.join();
    std::cout << "[Main] All consumers finished. Completed="
              << g_completed.size() << "\n";

    auto wall_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> wall_dur = wall_end - wall_start;
    double wall_time_s = wall_dur.count();

    printReport(cfg, wall_time_s);

    if (static_cast<int>(g_completed.size()) != cfg.total_requests) {
        std::cerr << "ERROR: lost "
                  << (cfg.total_requests - (int)g_completed.size())
                  << " requests!\n";
        return 1;
    }
    return 0;
}
