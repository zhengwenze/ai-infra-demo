#ifndef MINI_SCHEDULER_REQUEST_H
#define MINI_SCHEDULER_REQUEST_H

#include <string>

namespace scheduler {

// ============================================================
// Request：LLM 推理请求
// ============================================================

struct Request {
  int id;
  std::string prompt;
  int max_tokens; // 预期输出 token 数（用于模拟推理耗时）

  // 统计字段（由引擎层填写）
  double arrive_time_ms = 0.0; // 入队时刻（相对 epoch，ms）
  double start_time_ms = 0.0;  // 开始推理时刻（ms）
  double finish_time_ms = 0.0; // 推理完成时刻（ms）
  double latency_ms = 0.0;     // E2E 延迟 = finish - arrive

  Request() : id(0), max_tokens(128) {}

  Request(int id_, std::string prompt_, int max_tokens_)
      : id(id_), prompt(std::move(prompt_)), max_tokens(max_tokens_) {}
};

} // namespace scheduler

#endif // MINI_SCHEDULER_REQUEST_H
