#ifndef RTS_2_TESTEXCEPTION_H
#define RTS_2_TESTEXCEPTION_H

#include <stdexcept>

class TestException: public std::runtime_error
{
public:
    TestException(std::string const& msg);
};

#endif //RTS_2_TESTEXCEPTION_H
