#ifndef RTS_2_SRC_STRUCTURES_THREAD_POOL_H_
#define RTS_2_SRC_STRUCTURES_THREAD_POOL_H_

#include <chrono>
#include <cstring> // memcpy
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "../params/params.h"
#include "../probe/probe.h"
#include "pipeline.h"

template<class T>
class PipelineThreadPool {
 public:
  explicit PipelineThreadPool(Params &params, Probe &probe, uint32_t n_threads);
  ~PipelineThreadPool();

  void
  BlockEnqueueData(std::shared_ptr<T[]> buf, uint32_t buf_size,
                   uint64_t frame_offset);

  void StopWaiting() { wait_for_data = false; };

  // getters
  [[nodiscard]] bool
  is_working() const { return (wait_for_data || !work_queue.empty()); };
 private:
  Params params_;
  Probe probe_;

  std::vector<std::thread> threads;
  std::queue<Pipeline<T>> work_queue;

  std::mutex mutex_;

  uint32_t max_queue_size;
  bool wait_for_data = true;
};

#endif //RTS_2_SRC_STRUCTURES_THREAD_POOL_H_
