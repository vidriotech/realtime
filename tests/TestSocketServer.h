#ifndef RTS_2_TESTSOCKETSERVER_H
#define RTS_2_TESTSOCKETSERVER_H

#include <chrono> // std::chrono::seconds
#include <iostream>
#include <fstream>
#include <string>
#include <thread> // std::this_thread::sleep_for

#include<winsock2.h>
#pragma comment (lib, "Ws2_32.lib")

// 385 channels * 30000 samples/sec, plus terminating null character
#define TEST_BUFFER_SIZE 23100000
#define SOCKET_PORT 23905

int emulate_socket_server(const std::string& file_path, int n_loops);

#endif //RTS_2_TESTSOCKETSERVER_H
