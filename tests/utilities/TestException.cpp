#include "TestException.h"

TestException::TestException(const std::string &msg)
        : std::runtime_error(msg){};