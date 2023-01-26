#pragma once

namespace cf { // begin of namespace cf ===================================

class ThreadPool {

  friend class Scheduler;

  public:

    ThreadPool(size_t num_workers);
    ~ThreadPool();

    template <typename C, typename ...Args>
    auto enqueue (C&& callable, Args&&... args); 

  private:

    std::vector<std::thread> _workers;
    std::queue<std::function<void()> > _jobs;

    std::mutex _jobs_mutex;
    std::condition_variable cv;
    bool _stop;
};

inline
ThreadPool::ThreadPool(size_t num_workers)
: _stop(false)
{
  _workers.reserve(num_workers);
  for(size_t i=0; i<num_workers; ++i){
    _workers.emplace_back(
        [this] {
          while(true){
            std::function<void()> job;
            {
              std::unique_lock<std::mutex> lock(_jobs_mutex);
              cv.wait(lock, [this]{return this->_stop || (!this->_jobs.empty());});
              if(_stop && _jobs.empty()){
                return;
              }
              job = std::move(this->_jobs.front());
              this->_jobs.pop();
            }
            job();
          }
        }
    );
  }
}

template<typename C, typename ...Args>
auto ThreadPool::enqueue(C&& callable, Args&&... args){
  using return_type = typename std::result_of<C(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()> >(
      std::bind(std::forward<C>(callable), std::forward<Args>(args)...)
  );

  auto result =  (*task).get_future();
  {
    std::unique_lock<std::mutex> lock(_jobs_mutex);
    if(_stop){
      throw std::runtime_error("enqueueing to stopped ThreadPool");
    }
    _jobs.emplace([task]() { (*task)(); });

  }
  cv.notify_one();

  return result;
}

inline
ThreadPool::~ThreadPool(){
  {
    std::unique_lock<std::mutex> lock(_jobs_mutex);
    _stop = true;
  }
  cv.notify_all();
  for(auto &worker:_workers){
    worker.join();
  }

}

} // end of namespace coro ==============================================
