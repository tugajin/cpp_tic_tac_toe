#ifndef __THREAD_HPP__
#define __THREAD_HPP__
#include <mutex>

class Lockable {

protected : 

   mutable std::mutex mutex;

public :
   Lockable(){}
   Lockable(Lockable && /*lock*/) {
   }
   void lock () {
    this->mutex.lock();
   }
   void unlock () {
    this->mutex.unlock();
   }
};

#endif
