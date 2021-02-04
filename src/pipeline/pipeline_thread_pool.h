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
  explicit PipelineThreadPool(Params &params, Probe &probe, uint32_t n_threads)
      : params_(params), probe_(probe) {
    auto n_threads_available = std::thread::hardware_concurrency();

    n_threads = std::max(n_threads, (uint32_t) 1);
    n_threads = std::min(n_threads, n_threads_available);
    max_queue_size = 2 * n_threads;

    for (auto i = 0; i < n_threads; i++) {
      threads.template emplace_back(std::thread([this]() {
        while (true) {
          if (!mutex_.try_lock()) {
            continue;
          }

          if (!wait_for_data && work_queue.empty()) {
            mutex_.unlock();
            break;
          } else if (work_queue.empty()) {
            mutex_.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            continue;
          }

          auto pipeline = work_queue.front();
          work_queue.pop();

          auto tid = std::this_thread::get_id();
          std::cout << "thread " << tid << " has offset " <<
                    pipeline.frame_offset() << std::endl;
          mutex_.unlock();

          pipeline.Process();
        }
      }));
    }
  };
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
