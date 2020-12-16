#include <iostream>
#include <filesystem>
#include <thread>

#include "SocketReader.h"
#include "tests/TestSocketServer.h"

#define TEST_FILE R"(C:\Users\alanc\Dropbox (Vidrio Technologies)\rts-data\cortex-20160727\Hopkins_20160722_g0_t0.imec.ap_CAR.bin)"

using namespace std;
using namespace std::filesystem;

int main()
{
    char *buf = (char *)malloc(TEST_BUFFER_SIZE);
    if (buf) {
    std::thread server_thread(emulate_socket_server, TEST_FILE, 0);
    SocketReader<short> reader{};
    reader.connect(SOCKET_PORT);
//    path file_path = path(R"(C:\Users\alanc\Dropbox (Vidrio Technologies)\rts-data\colonell-20180919\concatenated.imec.ap.bin)");
    reader.read(buf, TEST_BUFFER_SIZE);
    std::cout << buf << std::endl;
    std::cout << "Hello, World!" << std::endl;
    free(buf);
    }
//    std::cout << file_path << std::endl;
//
//    std::vector<int> foo(100);
//
//    std::cout << foo.size() << std::endl;
//
//    foo.push_back(1);
//
//    for (int& i : foo) {
//        std::cout << i << " ";
//    }
//    std::cout << endl;
//
//    int bar[] = {1, 2, 3, 4};
//    foo.assign(bar, bar+4);
//
//    std::cout << foo.size() << std::endl;
//
//    for (int& i : foo) {
//        std::cout << i << " ";
//    }
//    std::cout << endl;
}
