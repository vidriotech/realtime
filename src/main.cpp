#include <iostream>
#include <filesystem>
#include <thread>

using namespace std;
using namespace std::filesystem;

std::string get_env_var(std::string const &key)
{
    char *val = getenv(key.c_str());
    return val == nullptr ? std::string("") : std::string(val);
}

int main()
{
    std::string test_file = get_env_var("TEST_FILE");
    std::cout << "Your test file is: " << test_file << std::endl;
}
