#ifdef _IA64_
#pragma warning (disable: 4311)
    #pragma warning (disable: 4312)
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif


#include <winsock2.h>
#include <ws2tcpip.h>
#include <mstcpip.h>
#include <stdio.h>


#define ERR(e) \
        printf("%s:%s failed: %d [%s@%ld]\n",__FUNCTION__,e,WSAGetLastError(),__FILE__,__LINE__)

#define CLOSESOCK(s) \
        if(INVALID_SOCKET != s) {closesocket(s); s = INVALID_SOCKET;}

#define DEFAULT_WAIT    30000

#define WS_VER          0x0202

#define DEFAULT_PORT    12345

#define TST_MSG         "0123456789abcdefghijklmnopqrstuvwxyz\0"

HANDLE hCloseSignal = NULL;

DWORD WINAPI ConnectThread(LPVOID pParam)
{
    pollfd fdarray = {0};
    SOCKET csock = INVALID_SOCKET;
    sockaddr_storage addrLoopback = {0};
    int ret = 0;
    unsigned long non_blocking_mode = 1;
    char buf[MAX_PATH] = {0};

    UNREFERENCED_PARAMETER(pParam);

    __try
    {
        if (INVALID_SOCKET == (csock = socket(AF_INET6,
                                              SOCK_STREAM,
                                              0
        )))
        {
            ERR("socket");
            __leave;
        }

        if (SOCKET_ERROR == ioctlsocket(csock,
                                        FIONBIO,
                                        &non_blocking_mode
        ))
        {
            ERR("FIONBIO");
            __leave;
        }

        addrLoopback.ss_family = AF_INET6;
        INETADDR_SETLOOPBACK((SOCKADDR*)&addrLoopback);
        SS_PORT((SOCKADDR*)&addrLoopback) = htons(DEFAULT_PORT);

        if (SOCKET_ERROR == connect(csock,
                                    (SOCKADDR*)&addrLoopback,
                                    sizeof (addrLoopback)
        ))
        {
            if (WSAEWOULDBLOCK != WSAGetLastError())
            {
                ERR("connect");
                __leave;
            }
        }

        //Call WSAPoll for writeability on connecting socket
        fdarray.fd = csock;
        fdarray.events = POLLWRNORM;

        if (SOCKET_ERROR == (ret = WSAPoll(&fdarray,
                                           1,
                                           DEFAULT_WAIT
        )))
        {
            ERR("WSAPoll");
            __leave;
        }

        if (ret)
        {
            if (fdarray.revents & POLLWRNORM)
            {
                printf("ConnectThread: Established connection\n");

                //Send data

                if (SOCKET_ERROR == (ret = send(csock,
                                                TST_MSG,
                                                sizeof (TST_MSG),
                                                0
                )))
                {
                    ERR("send");
                    __leave;
                }
                else
                    printf("ConnectThread: sent %d bytes\n",ret);

            }
        }


        //Call WSAPoll for readability on connected socket
        fdarray.events = POLLRDNORM;

        if (SOCKET_ERROR == (ret = WSAPoll(&fdarray,
                                           1,
                                           DEFAULT_WAIT
        )))
        {
            ERR("WSAPoll");
            __leave;
        }

        if (ret)
        {
            if (fdarray.revents & POLLRDNORM)
            {
                if (SOCKET_ERROR == (ret = recv(csock,
                                                buf,
                                                sizeof (buf),
                                                0
                )))
                {
                    ERR("recv");
                    __leave;
                }
                else
                    printf("ConnectThread: recvd %d bytes\n",ret);
            }
        }

        WaitForSingleObject(hCloseSignal,DEFAULT_WAIT);


    }
    __finally
    {
        CLOSESOCK(csock);
    }

    return 0;
}

int __cdecl main()
{
    WSAData wsd;
    int nStartup = 0;
    int nErr = 0;
    int ret = 0;

    SOCKET lsock = INVALID_SOCKET;
    SOCKET asock = INVALID_SOCKET;

    sockaddr_storage addr = {0};
    pollfd fdarray = {0};
    
    unsigned long non_blocking_mode = 1;
    
    char buf[MAX_PATH] = {0};
    
    HANDLE thread_handle = NULL;
    
    unsigned long dwThreadId = 0;


    __try
    {
        nErr = WSAStartup(WS_VER,&wsd);
        if (nErr)
        {
            WSASetLastError(nErr);
            ERR("WSAStartup");
            __leave;
        } else
            nStartup++;

        if (NULL == (hCloseSignal = CreateEvent(NULL,
                                                TRUE,
                                                FALSE,
                                                NULL
        )))
        {
            ERR("CreateEvent");
            __leave;
        }

        if (NULL == (thread_handle = CreateThread(NULL,
                                            0,
                                            ConnectThread,
                                            NULL,
                                            0,
                                            &dwThreadId
        )))
        {
            ERR("CreateThread");
            __leave;
        }

        addr.ss_family = AF_INET6;
        INETADDR_SETANY((SOCKADDR*)&addr);
        SS_PORT((SOCKADDR*)&addr) = htons(DEFAULT_PORT);

        if (INVALID_SOCKET == (lsock = socket(AF_INET6,
                                              SOCK_STREAM,
                                              0
        )))
        {
            ERR("socket");
            __leave;
        }

        if (SOCKET_ERROR == ioctlsocket(lsock,
                                        FIONBIO,
                                        &non_blocking_mode
        ))
        {
            ERR("FIONBIO");
            __leave;
        }


        if (SOCKET_ERROR == bind(lsock,
                                 (SOCKADDR*)&addr,
                                 sizeof (addr)
        ))
        {
            ERR("bind");
            __leave;
        }

        if (SOCKET_ERROR == listen(lsock,1))
        {
            ERR("listen");
            __leave;
        }

        //Call WSAPoll for readability of listener (accepted)

        fdarray.fd = lsock;
        fdarray.events = POLLRDNORM;

        if (SOCKET_ERROR == (ret = WSAPoll(&fdarray,
                                           1,
                                           DEFAULT_WAIT
        )))
        {
            ERR("WSAPoll");
            __leave;
        }

        if (ret)
        {
            if (fdarray.revents & POLLRDNORM)
            {
                printf("Main: Connection established.\n");

                if (INVALID_SOCKET == (asock = accept(lsock,
                                                      NULL,
                                                      NULL
                )))
                {
                    ERR("accept");
                    __leave;
                }

                if (SOCKET_ERROR == (ret = recv(asock,
                                                buf,
                                                sizeof(buf),
                                                0
                )))
                {
                    ERR("recv");
                    __leave;
                } else
                    printf("Main: recvd %d bytes\n",ret);
            }
        }

        //Call WSAPoll for writeability of accepted

        fdarray.fd = asock;
        fdarray.events = POLLWRNORM;

        if (SOCKET_ERROR == (ret = WSAPoll(&fdarray,
                                           1,
                                           DEFAULT_WAIT
        )))
        {
            ERR("WSAPoll");
            __leave;
        }

        if (ret)
        {
            if (fdarray.revents & POLLWRNORM)
            {
                if (SOCKET_ERROR == (ret = send(asock,
                                                TST_MSG,
                                                sizeof(TST_MSG),
                                                0
                )))
                {
                    ERR("send");
                    __leave;
                } else
                    printf("Main: sent %d bytes\n",ret);
            }
        }

        SetEvent(hCloseSignal);

        WaitForSingleObject(thread_handle,DEFAULT_WAIT);

    }
    __finally
    {
        CloseHandle(hCloseSignal);
        CloseHandle(thread_handle);
        CLOSESOCK(asock);
        CLOSESOCK(lsock);
        if(nStartup)
            WSACleanup();
    }

    return 0;

}