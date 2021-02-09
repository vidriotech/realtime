#include "pipeline_thread_pool.h"

template<class T>
PipelineThreadPool<T>::PipelineThreadPool(Params &params, Probe &probe,
                                          uint32_t n_threads)
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
}

template<class T>
PipelineThreadPool<T>::~PipelineThreadPool() {
  wait_for_data = false;

  for (auto &thread : threads)
    thread.join();
}

/**
 * @brief
 * @tparam T
 * @param buf
 * @param buf_size
 * @param frame_offset
 */
template<class T>
void PipelineThreadPool<T>::BlockEnqueueData(std::shared_ptr<T[]> buf,
                                             uint32_t buf_size,
                                             uint64_t frame_offset) {
  std::shared_ptr<T[]> buf_copy(new T[buf_size]);
  std::memcpy(buf_copy.get(), buf.get(), buf_size * sizeof(T));

  // block until a position in the queue becomes available
  while (work_queue.size() >= max_queue_size) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  Pipeline<T> pipeline(params_, probe_);
  pipeline.Update(buf_copy, buf_size, frame_offset);

  work_queue.push(pipeline);
}

template
class PipelineThreadPool<short>;
