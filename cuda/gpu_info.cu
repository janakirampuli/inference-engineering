#include "utils/cuda_utils.cuh"

#include <iomanip>
#include <iostream>

using namespace std;

int getAttr(cudaDeviceAttr attr, int dev)
{
    int value;
    CUDA_CHECK(cudaDeviceGetAttribute(&value, attr, dev));
    return value;
}

int fp32CoresPerSM(int major, int minor)
{
    if (major == 7 || (major == 8 && minor == 0)) return 64;
    if (major >= 8) return 128;
    return 0;
}

void printBasic(cudaDeviceProp &p, int dev)
{
    cout << "\n================ BASIC DEVICE INFO ================\n";

    cout << "Name                       : " << p.name << "\n";
    cout << "Compute Capability         : " << p.major << "." << p.minor << "\n";

    cout << "\n[Execution]\n";
    cout << "Multiprocessors (SMs)      : " << p.multiProcessorCount << "\n";
    cout << "SM clock (max)             : "
              << getAttr(cudaDevAttrClockRate, dev) / 1000 << " MHz\n";
    cout << "Warp Size                  : " << p.warpSize << "\n";
    cout << "Max threads per block      : " << p.maxThreadsPerBlock << "\n";
    cout << "Max threads dim            : "
              << p.maxThreadsDim[0] << " x "
              << p.maxThreadsDim[1] << " x "
              << p.maxThreadsDim[2] << "\n";
    cout << "Max grid size              : "
              << p.maxGridSize[0] << " x "
              << p.maxGridSize[1] << " x "
              << p.maxGridSize[2] << "\n";

    cout << "\n[Memory]\n";
    cout << "Total global memory        : "
              << fixed << setprecision(2)
              << p.totalGlobalMem / 1e9 << " GB\n";

    cout << "Shared memory per block    : "
              << p.sharedMemPerBlock / 1024.0 << " KB\n";

    cout << "Registers per block        : " << p.regsPerBlock << "\n";

    cout << "L2 Cache Size              : "
              << p.l2CacheSize / (1024.0*1024) << " MB\n";

    cout << "Memory clock rate          : "
              << getAttr(cudaDevAttrMemoryClockRate, dev) / 1000.0 << " MHz\n";

    cout << "Memory bus width           : "
              << p.memoryBusWidth << " bits\n";

    cout << "Async copy engines         : " << p.asyncEngineCount << "\n";
}

void printLowLevel(int dev)
{
    cout << "\n============== LOW-LEVEL ARCHITECTURE ==============\n";

    cout << "Max threads per SM             : "
              << getAttr(cudaDevAttrMaxThreadsPerMultiProcessor, dev) << "\n";

    cout << "Max blocks per SM              : "
              << getAttr(cudaDevAttrMaxBlocksPerMultiprocessor, dev) << "\n";

    // Computed dynamically as (MaxThreadsPerMultiProcessor / WarpSize)
    // because cudaDevAttrMaxWarpsPerMultiprocessor is not a native CUDA enum
    int maxThreads = getAttr(cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    int warpSize = getAttr(cudaDevAttrWarpSize, dev);
    cout << "Max warps per SM               : "
              << (maxThreads / warpSize) << "\n";

    cout << "Registers per SM               : "
              << getAttr(cudaDevAttrMaxRegistersPerMultiprocessor, dev) << "\n";

    cout << "Shared mem per SM              : "
              << getAttr(cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev) / 1024.0
              << " KB\n";

    cout << "Shared mem per block (opt-in)  : "
              << getAttr(cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) / 1024.0
              << " KB\n";

    cout << "Runtime shared mem overhead    : "
              << getAttr(cudaDevAttrReservedSharedMemoryPerBlock, dev)
              << " B\n";
}

void printPeak(cudaDeviceProp &p, int dev)
{
    cout << "\n=================== PEAK NUMBERS ===================\n";

    int sms = p.multiProcessorCount;
    cout << "Threads resident at once       : "
              << sms * getAttr(cudaDevAttrMaxThreadsPerMultiProcessor, dev) << "\n";

    double bandwidth =
        2.0 * getAttr(cudaDevAttrMemoryClockRate, dev) * (p.memoryBusWidth/8) / 1e6;
    cout << "Memory bandwidth               : " << bandwidth << " GB/s\n";

    int cores = fp32CoresPerSM(p.major, p.minor);
    if (cores == 0) {
        cout << "FP32 cores per SM              : unknown for this GPU\n";
        return;
    }

    double tflops = 2.0 * sms * cores * getAttr(cudaDevAttrClockRate, dev) / 1e9;
    cout << "FP32 cores per SM              : " << cores << "\n";
    cout << "FP32 math (FMA = 2 ops)        : " << tflops << " TFLOPS\n";
    cout << "FP32 ops per float loaded      : "
              << tflops * 1e12 / (bandwidth * 1e9 / 4) << "\n";
}

int main()
{
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        cout << "No CUDA devices found.\n";
        return 0;
    }

    cout << "Detected " << deviceCount << " CUDA device(s)\n";

    for (int dev = 0; dev < deviceCount; dev++) {

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        cout << "\n====================================================\n";
        cout << "DEVICE " << dev << "\n";
        cout << "====================================================\n";

        printBasic(prop, dev);
        printLowLevel(dev);
        printPeak(prop, dev);
    }

    return 0;
}
