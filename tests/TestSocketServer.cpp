#ifdef WIN32

#include "TestSocketServer.h"

class TestSocketServer {
private:
    SOCKET listen_socket;
    SOCKET client_socket;

    struct sockaddr_in server;
    struct sockaddr_in client;

    std::ifstream stream;
public:
    TestSocketServer(const std::string& file_path);

    void bind(int port);
};

TestSocketServer::TestSocketServer(const std::string& file_path)
    : stream(file_path, std::ifstream::in)
{
    if (!stream) {
        throw std::runtime_error("Failed to open file '" + file_path + "' for reading." );
    }

    WSADATA wsa;
    int c, res, bytes_sent;

    std::cout << "Initialising Winsock...";
    if (0 != WSAStartup(MAKEWORD(2,2), &wsa)) {
        WSACleanup();
        throw std::runtime_error("Failed. Error Code: " + std::to_string(WSAGetLastError()));
    }

    std::cout << "done." << std::endl;

    // create a socket
    if (INVALID_SOCKET == (listen_socket = socket(AF_INET, SOCK_STREAM, 0 ))) {
        WSACleanup();
        throw std::runtime_error("Could not create socket: " + std::to_string(WSAGetLastError() ));
    }

    std::cout << "Socket created." << std::endl;
}

void TestSocketServer::bind(int port)
{
    // prepare the sockaddr_in structure
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons(port);

    // bind
    if (SOCKET_ERROR == ::bind(listen_socket, (struct sockaddr *)&server, sizeof(server))) {
        closesocket(listen_socket);
        WSACleanup();
        throw std::runtime_error("Bind failed with error code: " + std::to_string(WSAGetLastError()));
    }

    // listen for incoming connections
    listen(listen_socket, 3);

    puts("Waiting for incoming connections...");
    // accept an incoming connection
}

int emulate_socket_server(const std::string& file_path, int n_loops)
{
    c = sizeof(struct sockaddr_in);
    client_socket = accept(listen_socket, (struct sockaddr *)&client, &c);
    if (INVALID_SOCKET == client_socket) {
        std::cout << "accept failed with error code: " << WSAGetLastError() << std::endl;
        closesocket(listen_socket);
        WSACleanup();
        return 1;
    }

    // create an ifstream from the file
    std::ifstream stream(file_path, std::ifstream::in);

    char *buf = (char *)malloc(TEST_BUFFER_SIZE);

    int loop = 0;
    // Receive until the peer shuts down the connection
    while (stream && loop < n_loops) {
        // Read data from the underlying file stream
        stream.read(buf, TEST_BUFFER_SIZE);
        bytes_sent = send(client_socket, buf, TEST_BUFFER_SIZE, 0 );
        if (bytes_sent == SOCKET_ERROR) {
            throw std::runtime_error("send failed with error: " + std::to_string(WSAGetLastError() ));
            closesocket(client_socket);
            WSACleanup();
            free(buf);
            return 1;
        }

        std::cout << "Bytes sent: " << bytes_sent << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        loop++;
    };

    // close the file stream, shutdown the connection
    stream.close();
    res = shutdown(client_socket, SD_SEND);
    if (res == SOCKET_ERROR) {
        throw std::runtime_error("shutdown failed with error: " + std::to_string(WSAGetLastError()));
        closesocket(client_socket);
        WSACleanup();
        free(buf);
        return 1;
    }

    // cleanup
    closesocket(client_socket);
    WSACleanup();

    free(buf);
    return 0;
}

#endif
