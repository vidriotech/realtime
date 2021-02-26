# Spiegel RTS

Spiegel RTS is software for real-time spike sorting.

## Building

This project builds with CMake.
You should have CUDA and its supported C++ compiler installed.
For example, we have tested with CUDA 10.2, which, on Linux, supports g++-8. 

The following commands should build:

```shell
cmake -B /path/to/build/dir
cd /path/to/build/dir
make -j8
```

You may have to manually specify your C++ compiler with, e.g., `-DCMAKE_CUDA_FLAGS=-ccbin=g++-8`.

In order to run the main program (which at the moment is a test program), you will need to have some environment variables defined.
See tests/README.md for more information.

## Testing

See tests/README.md for test preconditions.
Once you have built the program, you should see a `tests` folder in the build directory.
Simply run `rtstests` in that directory.
