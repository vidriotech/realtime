#ifndef RTS_2_TESTEXCEPTION_H
#define RTS_2_TESTEXCEPTION_H

#include <stdexcept>

class TestException : public std::runtime_error {
 public:
  explicit TestException(std::string const &msg)
      : std::runtime_error(msg) {};;
};

#endif //RTS_2_TESTEXCEPTION_H
