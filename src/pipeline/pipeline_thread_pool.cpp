#include "pipeline_thread_pool.h"

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
  std::shared_ptr<T[]> new_buf(new T[buf_size]);
  std::memcpy(new_buf.get(), buf.get(), buf_size * sizeof(T));

  // block until a position in the queue becomes available
  while (work_queue.size() >= max_queue_size) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  Pipeline<T> pipeline(params_, probe_);
  pipeline.Update(buf, buf_size, frame_offset);

  work_queue.push(pipeline);
}

template
class PipelineThreadPool<short>;
