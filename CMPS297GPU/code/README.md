# CUDA Code Samples

This directory contains various CUDA code samples and benchmarks.

## Build Instructions

You can use CMake to build all samples:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release -j8
```

## Running Tests

After building, you can run all tests using CTest:

```bash
ctest -C Release --output-on-failure
```

## GitHub CI/CD

This repository is configured with a GitHub Actions workflow for self-hosted runners.
The CI will:

1. Build all samples using CMake.
2. Run `query_device.gemini` to verify CUDA environment.
3. Run all other sample benchmarks/tests.

If no CUDA device is detected by the runner, the CI will fail.
