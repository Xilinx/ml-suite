#ifndef XDNN_UTIL_H
#define XDNN_UTIL_H

#include <chrono>
#include <unistd.h>

#define LOCATION "[ " << __FILE__ << ":" << __LINE__ << " ] "
#define HEX(n) "0x" << std::hex << n << std::dec

namespace xdnn {

class XTimer {
public:
  XTimer() :
      beg_(clock_::now()) {
  }
  void reset() {
    beg_ = clock_::now();
  }
  double elapsed() const {
    return std::chrono::duration_cast < second_ > (clock_::now() - beg_).count();
  }

private:
  typedef std::chrono::high_resolution_clock clock_;
  typedef std::chrono::duration<double, std::ratio<1> > second_;
  std::chrono::time_point<clock_> beg_;
};

class XDebugJournal {
public:
  XDebugJournal();
  void log(std::string key, std::string entry = "");
  template<class T>
  void logBlob(std::string key, T* blob, int count, int numToPrint);
  void logRegWrite(std::string key, int cuIdx, size_t offset, size_t size,
      const void *data);
  void logRegRead(std::string key, int cuIdx, size_t offset, size_t size,
      const void *data);

private:
  static int call_counter;
};

template <typename DType>
class XAlignedBlob {
public:
  XAlignedBlob() :
      _sz(0), _blob(NULL) {
  }
  XAlignedBlob(size_t numelem) :
      _sz(numelem), _blob(NULL) {
    reset(numelem);
  }
  ~XAlignedBlob() {
    if (_blob)
      free(_blob);
  }

  void *get() {
    return _blob;
  }
  size_t numelem() const {
      return _sz;
  }
  size_t sizeInBytes() const {
    return _sz * sizeof(DType);
  }

  void reset(size_t numelem) {
    if (numelem != _sz){
        if (_blob)
          free(_blob);
        _sz = numelem;
        _blob = NULL;
        if (posix_memalign((void**)&_blob, getpagesize(), _sz * sizeof(DType)))
          throw std::bad_alloc();
    }
  }

private:
  size_t _sz;
  DType *_blob;
};

extern "C" {
const char* XDNNGetHostDeviceName(const char *mode = NULL);
} // extern "C" 


} // namespace 'xdnn'

#endif
