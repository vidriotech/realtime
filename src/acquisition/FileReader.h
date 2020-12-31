//
// Created by alan on 12/21/20.
//

#ifndef RTS_2_FILEREADER_H
#define RTS_2_FILEREADER_H

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include "../Probe.h"

template <class T>
class FileReader {
public:
    explicit FileReader(std::string&, Probe&);
    ~FileReader();

    void acquire_frames(int, int, T*);

    std::string get_filename() const;
    unsigned long get_nframes() const;
private:
    std::string filename;
    Probe probe;
    std::ifstream fp;

    unsigned long long fsize;

    void open();
    void close();
};

template <class T>
FileReader<T>::FileReader(std::string& filename, Probe& probe)
    : filename(filename), probe(probe)
{
    //
    open();

    fp.seekg(0, std::ios::end);
    fsize = fp.tellg();

    close();
}

template<class T>
FileReader<T>::~FileReader()
{
    fp.close(); // no-op if already closed
}

template<class T>
void FileReader<T>::acquire_frames(int frame_offset, int nsamples, T *buf) {
    open(); // no-op if already open
    fp.seekg(frame_offset * probe.n_total() * sizeof(T), std::ios::beg);
    fp.read((char *) buf, sizeof(T) * nsamples);
}

template<class T>
unsigned long FileReader<T>::get_nframes() const {
    return fsize / (probe.n_total() * sizeof(T));
}

template<class T>
std::string FileReader<T>::get_filename() const {
    return filename;
}

template<class T>
void FileReader<T>::open()
{
    if (fp.is_open()) return;
    fp.open(filename, std::ios::in | std::ios::binary);
}

template<class T>
void FileReader<T>::close()
{
    if (!fp.is_open()) return;
    fp.close();
}

#endif //RTS_2_FILEREADER_H
