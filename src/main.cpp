#include "simd_dct.h"
#include "simd_platform.h"

#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#endif

//////////////////////////////////////////////////////////////////////////

size_t _RunCount = 32;

constexpr size_t MaxRunCount = 256;
static uint64_t _ClocksPerRun[MaxRunCount];
static uint64_t _NsPerRun[MaxRunCount];

//////////////////////////////////////////////////////////////////////////

uint64_t GetCurrentTimeTicks();
uint64_t TicksToNs(const uint64_t ticks);

//////////////////////////////////////////////////////////////////////////

void print_perf_info(const size_t fileSize)
{
  if (_RunCount > 1)
  {
    uint64_t completeNs = 0;
    uint64_t completeClocks = 0;
    uint64_t minNs = (uint64_t)-1;
    uint64_t minClocks = (uint64_t)-1;

    for (size_t i = 0; i < _RunCount; i++)
    {
      completeNs += _NsPerRun[i];
      completeClocks += _ClocksPerRun[i];

      if (_NsPerRun[i] < minNs)
        minNs = _NsPerRun[i];

      if (_ClocksPerRun[i] < minClocks)
        minClocks = _ClocksPerRun[i];
    }

    const double meanNs = completeNs / (double)_RunCount;
    const double meanClocks = completeClocks / (double)_RunCount;
    double stdDevNs = 0;
    double stdDevClocks = 0;

    for (size_t i = 0; i < _RunCount; i++)
    {
      const double diffNs = _NsPerRun[i] - meanNs;
      const double diffClocks = _ClocksPerRun[i] - meanClocks;

      stdDevNs += diffNs * diffNs;
      stdDevClocks += diffClocks * diffClocks;
    }

    stdDevNs = sqrt(stdDevNs / (double)(_RunCount - 1));
    stdDevClocks = sqrt(stdDevClocks / (double)(_RunCount - 1));

    printf("| %7.2f clk/byte | %7.2f clk/byte (%7.2f ~ %7.2f) ", minClocks / (double_t)fileSize, meanClocks / fileSize, (meanClocks - stdDevClocks) / fileSize, (meanClocks + stdDevClocks) / fileSize);
    printf("| %8.2f MiB/s | %8.2f MiB/s (%8.2f ~ %8.2f)\n", (fileSize / (1024.0 * 1024.0)) / (minNs * 1e-9), (fileSize / (1024.0 * 1024.0)) / (meanNs * 1e-9), (fileSize / (1024.0 * 1024.0)) / ((meanNs + stdDevNs) * 1e-9), (fileSize / (1024.0 * 1024.0)) / ((meanNs - stdDevNs) * 1e-9));
  }
  else
  {
    printf("| %7.2f clk/byte | %7.2f clk/byte (%7.2f ~ %7.2f) ", _ClocksPerRun[0] / (double_t)fileSize, _ClocksPerRun[0] / (double)fileSize, (_ClocksPerRun[0]) / (double)fileSize, (_ClocksPerRun[0]) / (double)fileSize);
    printf("| %8.2f MiB/s | %8.2f MiB/s (%8.2f ~ %8.2f)\n", (fileSize / (1024.0 * 1024.0)) / (_NsPerRun[0] * 1e-9), (fileSize / (1024.0 * 1024.0)) / (_NsPerRun[0] * 1e-9), (fileSize / (1024.0 * 1024.0)) / ((_NsPerRun[0]) * 1e-9), (fileSize / (1024.0 * 1024.0)) / ((_NsPerRun[0]) * 1e-9));
  }
}

//////////////////////////////////////////////////////////////////////////

int32_t main(int32_t argc, char **pArgv)
{
  if (argc < 4)
  {
    puts("Invalid Parameter.\n\nUsage: simd_dct <raw_grayscale_image_file> <resolutionX> <resolutionY> [quality_level [out_file_name]]");
    return 1;
  }

  const char *filename = pArgv[1];
  const char *outFilename = nullptr;
  float qualityLevel = 50;
  const size_t sizeX = strtoull(pArgv[2], nullptr, 10);
  const size_t sizeY = strtoull(pArgv[3], nullptr, 10);

  if (sizeX == 0 || sizeY == 0)
  {
    puts("Invalid Resolution Specified. Aborting.");
    return 1;
  }

  size_t fileSize = 0;
  uint8_t *pInData = nullptr;
  uint8_t *pOutData = nullptr;

  // Read File.
  {
    FILE *pFile = fopen(filename, "rb");

    if (!pFile)
    {
      puts("Failed to read file.");
      return 1;
    }

    fseek(pFile, 0, SEEK_END);
    fileSize = ftell(pFile);

    if (fileSize <= 0)
    {
      puts("Invalid File size / failed to read file.");
      fclose(pFile);
      return 1;
    }

    fseek(pFile, 0, SEEK_SET);

    pInData = reinterpret_cast<uint8_t *>(malloc(fileSize));
    pOutData = reinterpret_cast<uint8_t *>(malloc(fileSize));

    if (pInData == nullptr || pOutData == nullptr)
    {
      puts("Memory allocation failure.");
      fclose(pFile);
      return 1;
    }

    if (fileSize != fread(pInData, 1, fileSize, pFile))
    {
      puts("Failed to read file.");
      fclose(pFile);
      return 1;
    }

    fclose(pFile);
  }

  float quantizeBase[] =
  {
    .17f, .11f, .10f, .16f,  .24f,  .40f,  .51f,  .61f,
    .12f, .12f, .14f, .19f,  .26f,  .58f,  .60f,  .55f,
    .14f, .13f, .16f, .24f,  .40f,  .57f,  .69f,  .56f,
    .14f, .17f, .22f, .29f,  .51f,  .87f,  .80f,  .62f,
    .18f, .22f, .37f, .56f,  .68f, 1.09f, 1.03f,  .77f,
    .24f, .35f, .55f, .64f,  .81f, 1.04f, 1.13f,  .92f,
    .49f, .64f, .78f, .87f, 1.03f, 1.21f, 1.20f, 1.01f,
    .72f, .92f, .95f, .98f, 1.12f, 1.00f, 1.03f,  .99f
  };

  if (argc == 5)
    qualityLevel = (float)strtoull(pArgv[4], nullptr, 10);

  for (size_t i = 0; i < 64; i++)
    quantizeBase[i] *= qualityLevel;

  if (argc == 6)
    outFilename = pArgv[5];

  _DetectCPUFeatures();

  // Print info. 
  {
    printf("File: '%s' (%" PRIu64 " Bytes)\n", filename, fileSize);

    printf("CPU: '%s' [%s] (Family: 0x%" PRIX8 " / Model: 0x%" PRIX8 " (0x%" PRIX8 ") / Stepping: 0x%" PRIX8 ")\nFeatures:", _CpuName, _GetCPUArchitectureName(), _CpuFamily, _CpuModel, _CpuExtModel, _CpuStepping);

    bool anysse = false;
    bool anyavx512 = false;

#define PRINT_WITH_CPUFLAG(flag, name, b, desc) if (flag) { if (!b) { fputs(desc, stdout); b = true; } else { fputs("/", stdout); } fputs(name, stdout); }
#define PRINT_SSE(flag, name) PRINT_WITH_CPUFLAG(flag, name, anysse, " SSE");
#define PRINT_AVX512(flag, name) PRINT_WITH_CPUFLAG(flag, name, anyavx512, " AVX-512");

    PRINT_SSE(sse2Supported, "2");
    PRINT_SSE(sse3Supported, "3");
    PRINT_SSE(sse41Supported, "4.1");
    PRINT_SSE(sse42Supported, "4.2");

    if (ssse3Supported)
      fputs(" SSSE3", stdout);

    if (avx2Supported)
      fputs(" AVX", stdout);

    if (avx2Supported)
      fputs(" AVX2", stdout);

    PRINT_AVX512(avx512FSupported, "F");
    PRINT_AVX512(avx512PFSupported, "PF");
    PRINT_AVX512(avx512ERSupported, "ER");
    PRINT_AVX512(avx512CDSupported, "CD");
    PRINT_AVX512(avx512BWSupported, "BW");
    PRINT_AVX512(avx512DQSupported, "DQ");
    PRINT_AVX512(avx512VLSupported, "VL");
    PRINT_AVX512(avx512IFMASupported, "IFMA");
    PRINT_AVX512(avx512VBMISupported, "VBMI");
    PRINT_AVX512(avx512VNNISupported, "VNNI");
    PRINT_AVX512(avx512VBMI2Supported, "VBMI2");
    PRINT_AVX512(avx512POPCNTDQSupported, "POPCNTDQ");
    PRINT_AVX512(avx512BITALGSupported, "BITALG");
    PRINT_AVX512(avx5124VNNIWSupported, "4VNNIW");
    PRINT_AVX512(avx5124FMAPSSupported, "4FMAPS");

    if (fma3Supported)
      fputs(" FMA3", stdout);

    if (aesNiSupported)
      fputs(" AES-NI", stdout);

#undef PRINT_WITH_CPUFLAG
#undef PRINT_SSE
#undef PRINT_AVX512

    puts("\n");
  }

  //// EncodeQuantizeReorderStereo.
  //{
  //  for (size_t run = 0; run < _RunCount; run++)
  //  {
  //    const uint64_t startTick = GetCurrentTimeTicks();
  //    const uint64_t startClock = __rdtsc();
  //    const simdDctResult result = simdDCT_EncodeQuantizeReorderStereoBuffer(pInData, pOutData, quantizeBase, sizeX, sizeY, 0, sizeY);
  //    const uint64_t endClock = __rdtsc();
  //    const uint64_t endTick = GetCurrentTimeTicks();
  //
  //    _mm_mfence();
  //
  //    _NsPerRun[run] = TicksToNs(endTick - startTick);
  //    _ClocksPerRun[run] = endClock - startClock;
  //
  //    printf("\r%3" PRIu64 ": %6.3f clocks/byte, %5.2f MiB/s", run + 1, (endClock - startClock) / (double)fileSize, (fileSize / (1024.0 * 1024.0)) / (TicksToNs(endTick - startTick) * 1e-9));
  //
  //    if (_FAILED(result))
  //    {
  //      printf("Encode failed with error code 0x%" PRIX64 ". Aborting.\n", (uint64_t)result);
  //      return 1;
  //    }
  //  }
  //
  //  printf("\r%-30s ", "EncQuantReordStereo");
  //  print_perf_info(fileSize);
  //}
  //
  //// EncodeQuantize.
  //{
  //  for (size_t run = 0; run < _RunCount; run++)
  //  {
  //    const uint64_t startTick = GetCurrentTimeTicks();
  //    const uint64_t startClock = __rdtsc();
  //    const simdDctResult result = simdDCT_EncodeQuantizeBuffer(pInData, pOutData, quantizeBase, sizeX, sizeY, 0, sizeY);
  //    const uint64_t endClock = __rdtsc();
  //    const uint64_t endTick = GetCurrentTimeTicks();
  //
  //    _mm_mfence();
  //
  //    _NsPerRun[run] = TicksToNs(endTick - startTick);
  //    _ClocksPerRun[run] = endClock - startClock;
  //
  //    printf("\r%3" PRIu64 ": %6.3f clocks/byte, %5.2f MiB/s", run + 1, (endClock - startClock) / (double)fileSize, (fileSize / (1024.0 * 1024.0)) / (TicksToNs(endTick - startTick) * 1e-9));
  //
  //    if (_FAILED(result))
  //    {
  //      printf("Encode failed with error code 0x%" PRIX64 ". Aborting.\n", (uint64_t)result);
  //      return 1;
  //    }
  //  }
  //
  //  printf("\r%-30s ", "Encode Quantize");
  //  print_perf_info(fileSize);
  //}

  // EncodeQuantize32Reorder.
  {
    for (size_t run = 0; run < _RunCount; run++)
    {
      const uint64_t startTick = GetCurrentTimeTicks();
      const uint64_t startClock = __rdtsc();
      const simdDctResult result = simdDCT_EncodeQuantize32ReorderBuffer(pInData, pOutData, quantizeBase, sizeX, sizeY, 0, sizeY);
      const uint64_t endClock = __rdtsc();
      const uint64_t endTick = GetCurrentTimeTicks();

      _mm_mfence();

      _NsPerRun[run] = TicksToNs(endTick - startTick);
      _ClocksPerRun[run] = endClock - startClock;

      printf("\r%3" PRIu64 ": %6.3f clocks/byte, %5.2f MiB/s", run + 1, (endClock - startClock) / (double)fileSize, (fileSize / (1024.0 * 1024.0)) / (TicksToNs(endTick - startTick) * 1e-9));

      if (_FAILED(result))
      {
        printf("Encode failed with error code 0x%" PRIX64 ". Aborting.\n", (uint64_t)result);
        return 1;
      }
    }

    printf("\r%-30s ", "Encode Quantize 32 Reorder");
    print_perf_info(fileSize);
  }

  if (outFilename != nullptr)
  {
    FILE *pFile = fopen(outFilename, "wb");
    
    if (pFile == nullptr)
    {
      puts("Failed to open destination file. Aborting.\n");
      return 1;
    }

    fwrite(pOutData, 1, sizeX * sizeY, pFile);
    fclose(pFile);
  }

  return 0;
}

//////////////////////////////////////////////////////////////////////////

uint64_t GetCurrentTimeTicks()
{
#ifdef WIN32
  LARGE_INTEGER now;
  QueryPerformanceCounter(&now);

  return now.QuadPart;
#else
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);

  return (uint64_t)time.tv_sec * 1000000000 + (uint64_t)time.tv_nsec;
#endif
}

uint64_t TicksToNs(const uint64_t ticks)
{
#ifdef WIN32
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);

  return (ticks * 1000 * 1000 * 1000) / freq.QuadPart;
#else
  return ticks;
#endif
}
