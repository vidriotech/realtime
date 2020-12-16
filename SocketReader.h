#ifndef RTS_2_SOCKETREADER_H
#define RTS_2_SOCKETREADER_H

#ifndef SOCKET_PORT
#endif

#include <stdexcept>
#include <string>
#include<winsock2.h>
#include <ws2tcpip.h>
#pragma comment (lib, "Ws2_32.lib")

template <class T>
class SocketReader
{
private:
    SOCKET sock;
public:
    SocketReader();

    void close();
    void connect(int port);
    int read(char *buf, int buf_size);
};

template <class T>
SocketReader<T>::SocketReader()
{
    // initialize winsock
    WSADATA wsa;

    if (0 != WSAStartup(MAKEWORD(2,2), &wsa)) {
        WSACleanup();
        throw std::runtime_error("Failed. Error Code: " + std::to_string(WSAGetLastError()));
    }
}

template <class T>
void SocketReader<T>::close()
{
    if (INVALID_SOCKET != sock) {
        closesocket(sock);
    }
}

template <class T>
void SocketReader<T>::connect(int port)
{
    int res;

    struct addrinfo *result = nullptr,
            hints;

    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    // Resolve the server address and port
    res = getaddrinfo("127.0.0.1", std::to_string(port).c_str(), &hints, &result);
    if (res != 0) {
        WSACleanup();
        throw std::runtime_error("getaddrinfo failed with error: " + std::to_string(res));
    }

    // create a socket
    if (INVALID_SOCKET == (sock = socket(result->ai_family, result->ai_socktype, result->ai_protocol))) {
        WSACleanup();
        throw std::runtime_error("Could not create socket: " + std::to_string(WSAGetLastError()));
    }

    res = ::connect(sock, result->ai_addr, (int)result->ai_addrlen);
    if (res == SOCKET_ERROR) {
        closesocket(sock);
        sock = INVALID_SOCKET;
    }
}

template <class T>
int SocketReader<T>::read(char *buf, int buf_size) {
    int n_bytes_read = 0;
    if (INVALID_SOCKET != sock) {
        n_bytes_read = recv(sock, buf, buf_size, 0);
    }

    return n_bytes_read;
}

#endif //RTS_2_SOCKETREADER_H
