#include "simd_dct.h"
#include "simd_platform.h"

#include <cmath>
#include <algorithm>

// The following reverse macros are used so that int32_t/float array indices match the packed value positions.
#define _mm_set_epi32_reverse(a, b, c, d) _mm_set_epi32(d, c, b, a)
#define _mm_set_ps_reverse(a, b, c, d) _mm_set_ps(d, c, b, a)
#define _mm_set_epi8_reverse(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) _mm_set_epi8(p, o, n, m, l, k, j, i, h, g, f, e, d, c, b, a)
#define _MM_SHUFFLE_REVERSE(a, b, c, d) _MM_SHUFFLE(d, c, b, a)

#define _FAIL_INTERNAL(code) \
  do \
  { result = code; \
    goto epilogue; \
  } while (0)

#define _ERROR_CHECK(functionCall) \
  do \
  { if (_FAILED(result = (functionCall))) \
    { _FAIL_INTERNAL(result); \
    } \
  } while (0)

#define _ERROR_IF(booleanExpression, result) \
  do \
  { if (booleanExpression) \
      _FAIL_INTERNAL(result); \
  } while (0)

#ifdef _MSC_VER
#define _ALIGN(bytes) __declspec(align(bytes))
#else
#define _ALIGN(bytes) __attribute__((aligned(bytes)))
#endif

#if defined(_MSC_VER) || defined(__clang__)
#define _VECTORCALL __vectorcall
#else
#define _VECTORCALL
#endif

template <typename T>
inline T constexpr _min(T a, T b) { return (a < b) ? a : b; }

template <typename T>
inline T constexpr _max(T a, T b) { return (a > b) ? a : b; }

template <typename T, typename U, typename V>
inline constexpr T _clamp(const T value, const U min, const V max)
{
  return value > min ? (value < max ? value : (T)max) : (T)min;
}

void simdDCT_EncodeQuantizeReorderStereoBuffer_NoSimd_Float(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY);
void simdDCT_EncodeQuantizeReorderStereoBuffer_SSE41_Float(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY);
void simdDCT_EncodeQuantizeReorderStereoBuffer_SSE2_Float(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY);
void simdDCT_EncodeQuantizeReorderStereoBuffer_SSSE3_Float(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY);

void simdDCT_EncodeQuantizeBuffer_NoSimd_Float(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY);
void simdDCT_EncodeQuantizeBuffer_SSE41_Float(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY);
void simdDCT_EncodeQuantizeBuffer_SSE2_Float(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY);
void simdDCT_EncodeQuantizeBuffer_SSSE3_Float(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY);

//////////////////////////////////////////////////////////////////////////

simdDctResult simdDCT_EncodeQuantizeReorderStereoBuffer(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY)
{
  simdDctResult result = sdr_Success;

  _ERROR_IF(pFrom == nullptr || pTo == nullptr, sdr_InvalidParameter);
  _ERROR_IF((sizeX & ~7) != sizeX || (sizeY & ~7) != sizeY, sdr_NotSupported);

  if (sse41Supported && sse2Supported)
    simdDCT_EncodeQuantizeReorderStereoBuffer_SSE41_Float(pFrom, pTo, pQuantizeLUT, sizeX, sizeY, startY, endY);
  else if (ssse3Supported && sse2Supported)
    simdDCT_EncodeQuantizeReorderStereoBuffer_SSSE3_Float(pFrom, pTo, pQuantizeLUT, sizeX, sizeY, startY, endY);
  else if (sse2Supported)
    simdDCT_EncodeQuantizeReorderStereoBuffer_SSE2_Float(pFrom, pTo, pQuantizeLUT, sizeX, sizeY, startY, endY);
  else
    simdDCT_EncodeQuantizeReorderStereoBuffer_NoSimd_Float(pFrom, pTo, pQuantizeLUT, sizeX, sizeY, startY, endY);
  
  goto epilogue;

epilogue:
  return result;
}

simdDctResult simdDCT_EncodeQuantizeBuffer(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY)
{
  simdDctResult result = sdr_Success;

  _ERROR_IF(pFrom == nullptr || pTo == nullptr, sdr_InvalidParameter);
  _ERROR_IF((sizeX & ~7) != sizeX || (sizeY & ~7) != sizeY, sdr_NotSupported);

  if (sse41Supported && sse2Supported)
    simdDCT_EncodeQuantizeBuffer_SSE41_Float(pFrom, pTo, pQuantizeLUT, sizeX, sizeY, startY, endY);
  else if (ssse3Supported && sse2Supported)
    simdDCT_EncodeQuantizeBuffer_SSSE3_Float(pFrom, pTo, pQuantizeLUT, sizeX, sizeY, startY, endY);
  else
    simdDCT_EncodeQuantizeBuffer_NoSimd_Float(pFrom, pTo, pQuantizeLUT, sizeX, sizeY, startY, endY);

  goto epilogue;

epilogue:
  return result;
}

//////////////////////////////////////////////////////////////////////////

// Reference Implementation.
inline static void inplace_dct8(IN float *pBuffer)
{
  constexpr float C_a = 1.3870398453221474618216191915664f;  // sqrt(2) * cos(1 * pi / 16)
  constexpr float C_b = 1.3065629648763765278566431734272f;  // sqrt(2) * cos(2 * pi / 16)
  constexpr float C_c = 1.1758756024193587169744671046113f;  // sqrt(2) * cos(3 * pi / 16)
  constexpr float C_d = 0.78569495838710218127789736765722f; // sqrt(2) * cos(5 * pi / 16)
  constexpr float C_e = 0.54119610014619698439972320536639f; // sqrt(2) * cos(6 * pi / 16)
  constexpr float C_f = 0.27589937928294301233595756366937f; // sqrt(2) * cos(7 * pi / 16)
  constexpr float C_norm = 0.35355339059327376220042218105242f; // 1 / sqrt(8)    

  const float x07p = pBuffer[0] + pBuffer[7];
  const float x16p = pBuffer[1] + pBuffer[6];
  const float x25p = pBuffer[2] + pBuffer[5];
  const float x34p = pBuffer[3] + pBuffer[4];

  const float x07m = pBuffer[0] - pBuffer[7];
  const float x61m = pBuffer[6] - pBuffer[1];
  const float x25m = pBuffer[2] - pBuffer[5];
  const float x43m = pBuffer[4] - pBuffer[3];

  const float x07p34pp = x07p + x34p;
  const float x07p34pm = x07p - x34p;
  const float x16p25pp = x16p + x25p;
  const float x16p25pm = x16p - x25p;

  pBuffer[0] = C_norm * (x07p34pp + x16p25pp);
  pBuffer[2] = C_norm * (C_b * x07p34pm + C_e * x16p25pm);
  pBuffer[4] = C_norm * (x07p34pp - x16p25pp);
  pBuffer[6] = C_norm * (C_e * x07p34pm - C_b * x16p25pm);

  pBuffer[1] = C_norm * (C_a * x07m - C_c * x61m + C_d * x25m - C_f * x43m);
  pBuffer[3] = C_norm * (C_c * x07m + C_f * x61m - C_a * x25m + C_d * x43m);
  pBuffer[5] = C_norm * (C_d * x07m + C_a * x61m + C_f * x25m - C_c * x43m);
  pBuffer[7] = C_norm * (C_f * x07m + C_d * x61m + C_c * x25m + C_a * x43m);
}

//////////////////////////////////////////////////////////////////////////

// Reference Implementation.
void simdDCT_EncodeQuantizeReorderStereoBuffer_NoSimd_Float(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY)
{
  struct internal
  {
    union intBuffer_t
    {
      uint8_t u8[64];
      uint64_t u64[8];
    };

    static_assert(sizeof(intBuffer_t) == sizeof(uint8_t) * 64, "Invalid Struct Packing.");

    static inline void encode_line(const size_t sizeX, IN_OUT intBuffer_t *pIntBuffer, IN_OUT float fBuffer[64], IN const float *pQuantizeLUT, IN const uint8_t *pBlockStart, IN_OUT uint8_t *pOutPositions[64])
    {
      constexpr float vr = .95f;
      constexpr float subtract = (127.0f / 255.0f);

      const float qTable[64] =
      {
        1.f / (pQuantizeLUT[ 0] * vr), 1.f / (pQuantizeLUT[ 1] * vr), 1.f / (pQuantizeLUT[ 2] * vr), 1.f / (pQuantizeLUT[ 3] * vr),
        1.f / (pQuantizeLUT[ 4] * vr), 1.f / (pQuantizeLUT[ 5] * vr), 1.f / (pQuantizeLUT[ 6] * vr), 1.f / (pQuantizeLUT[ 7] * vr),
        1.f / (pQuantizeLUT[ 8] * vr), 1.f / (pQuantizeLUT[ 9] * vr), 1.f / (pQuantizeLUT[10] * vr), 1.f / (pQuantizeLUT[11] * vr),
        1.f / (pQuantizeLUT[12] * vr), 1.f / (pQuantizeLUT[13] * vr), 1.f / (pQuantizeLUT[14] * vr), 1.f / (pQuantizeLUT[15] * vr),
        1.f / (pQuantizeLUT[16] * vr), 1.f / (pQuantizeLUT[17] * vr), 1.f / (pQuantizeLUT[18] * vr), 1.f / (pQuantizeLUT[19] * vr),
        1.f / (pQuantizeLUT[20] * vr), 1.f / (pQuantizeLUT[21] * vr), 1.f / (pQuantizeLUT[22] * vr), 1.f / (pQuantizeLUT[23] * vr),
        1.f / (pQuantizeLUT[24] * vr), 1.f / (pQuantizeLUT[25] * vr), 1.f / (pQuantizeLUT[26] * vr), 1.f / (pQuantizeLUT[27] * vr),
        1.f / (pQuantizeLUT[28] * vr), 1.f / (pQuantizeLUT[29] * vr), 1.f / (pQuantizeLUT[30] * vr), 1.f / (pQuantizeLUT[31] * vr),
        1.f / (pQuantizeLUT[32] * vr), 1.f / (pQuantizeLUT[33] * vr), 1.f / (pQuantizeLUT[34] * vr), 1.f / (pQuantizeLUT[35] * vr),
        1.f / (pQuantizeLUT[36] * vr), 1.f / (pQuantizeLUT[37] * vr), 1.f / (pQuantizeLUT[38] * vr), 1.f / (pQuantizeLUT[39] * vr),
        1.f / (pQuantizeLUT[40] * vr), 1.f / (pQuantizeLUT[41] * vr), 1.f / (pQuantizeLUT[42] * vr), 1.f / (pQuantizeLUT[43] * vr),
        1.f / (pQuantizeLUT[44] * vr), 1.f / (pQuantizeLUT[45] * vr), 1.f / (pQuantizeLUT[46] * vr), 1.f / (pQuantizeLUT[47] * vr),
        1.f / (pQuantizeLUT[48] * vr), 1.f / (pQuantizeLUT[49] * vr), 1.f / (pQuantizeLUT[50] * vr), 1.f / (pQuantizeLUT[51] * vr),
        1.f / (pQuantizeLUT[52] * vr), 1.f / (pQuantizeLUT[53] * vr), 1.f / (pQuantizeLUT[54] * vr), 1.f / (pQuantizeLUT[55] * vr),
        1.f / (pQuantizeLUT[56] * vr), 1.f / (pQuantizeLUT[57] * vr), 1.f / (pQuantizeLUT[58] * vr), 1.f / (pQuantizeLUT[59] * vr),
        1.f / (pQuantizeLUT[60] * vr), 1.f / (pQuantizeLUT[61] * vr), 1.f / (pQuantizeLUT[62] * vr), 1.f / (pQuantizeLUT[63] * vr)
      };

      for (size_t x = 0; x < sizeX; x += 8)
      {
        // Acquire block.
        for (size_t i = 0; i < 8; i++)
          pIntBuffer->u64[i] = *reinterpret_cast<const uint64_t *>(pBlockStart + sizeX * i);

        // Convert to float.
        for (size_t i = 0; i < 64; i++)
          fBuffer[i] = pIntBuffer->u8[i] / 255.f;

        // Swap dimensions.
        for (size_t i = 0; i < 8; i++)
          for (size_t j = i + 1; j < 8; j++)
            std::swap(fBuffer[j + i * 8], fBuffer[i + j * 8]);

        // Apply discrete cosine transform.
        for (size_t i = 0; i < 8; i++)
          inplace_dct8(fBuffer + i * 8);

        // Swap dimensions.
        for (size_t i = 0; i < 8; i++)
          for (size_t j = i + 1; j < 8; j++)
            std::swap(fBuffer[j + i * 8], fBuffer[i + j * 8]);

        // Apply discrete cosine transform.
        for (size_t i = 0; i < 8; i++)
          inplace_dct8(fBuffer + i * 8);

        // Convert & Store.
        for (size_t i = 0; i < 64; i++)
        {
          *pOutPositions[i] = (uint8_t)roundf(_clamp((fBuffer[i] * qTable[i]) + subtract, 0.f, 1.f) * 255.f);
          ++pOutPositions[i];
        }

        pBlockStart += 8;
      }
    }
  };

  internal::intBuffer_t intBuffer;
  float fBuffer[64];
  uint8_t *pOutPositions[64];
  
  // Interleaving stereo buffer write positions.
  {
    const size_t componentOffset = (sizeX * sizeY) / 64;

    for (size_t i = 0; i < 64; i++)
      pOutPositions[i] = pTo + componentOffset * i;
  }

  const uint8_t *pLine = pFrom;

  for (size_t y = 0; y < sizeY / 2; y += 8)
  {
    if (y * 2 < startY)
    {
      pLine += 8 * sizeX;
      
      for (size_t i = 0; i < 64; i++)
        pOutPositions[i] += (sizeX / 4); // Two buffers.

      continue;
    }
    else if (y * 2 > endY)
    {
      break;
    }

    // Left eye.
    {
      const uint8_t *pBlockStart = pLine;
      internal::encode_line(sizeX, &intBuffer, fBuffer, pQuantizeLUT, pBlockStart, pOutPositions);
    }

    // Right eye.
    {
      const uint8_t *pBlockStart = pLine + (sizeX * sizeY / 2);
      internal::encode_line(sizeX, &intBuffer, fBuffer, pQuantizeLUT, pBlockStart, pOutPositions);
    }

    pLine += 8 * sizeX;
  }
}

void simdDCT_EncodeQuantizeBuffer_NoSimd_Float(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY)
{
  struct internal
  {
    union intBuffer_t
    {
      uint8_t u8[64];
      uint64_t u64[8];
    };

    static_assert(sizeof(intBuffer_t) == sizeof(uint8_t) * 64, "Invalid Struct Packing.");

    static inline void encode_line(const size_t sizeX, IN_OUT intBuffer_t *pIntBuffer, IN_OUT float fBuffer[64], IN const float *pQuantizeLUT, IN const uint8_t *pBlockStart, OUT uint8_t *pTarget)
    {
      constexpr float vr = .95f;
      constexpr float subtract = (127.0f / 255.0f);

      const float qTable[64] =
      {
        1.f / (pQuantizeLUT[0] * vr), 1.f / (pQuantizeLUT[1] * vr), 1.f / (pQuantizeLUT[2] * vr), 1.f / (pQuantizeLUT[3] * vr),
        1.f / (pQuantizeLUT[4] * vr), 1.f / (pQuantizeLUT[5] * vr), 1.f / (pQuantizeLUT[6] * vr), 1.f / (pQuantizeLUT[7] * vr),
        1.f / (pQuantizeLUT[8] * vr), 1.f / (pQuantizeLUT[9] * vr), 1.f / (pQuantizeLUT[10] * vr), 1.f / (pQuantizeLUT[11] * vr),
        1.f / (pQuantizeLUT[12] * vr), 1.f / (pQuantizeLUT[13] * vr), 1.f / (pQuantizeLUT[14] * vr), 1.f / (pQuantizeLUT[15] * vr),
        1.f / (pQuantizeLUT[16] * vr), 1.f / (pQuantizeLUT[17] * vr), 1.f / (pQuantizeLUT[18] * vr), 1.f / (pQuantizeLUT[19] * vr),
        1.f / (pQuantizeLUT[20] * vr), 1.f / (pQuantizeLUT[21] * vr), 1.f / (pQuantizeLUT[22] * vr), 1.f / (pQuantizeLUT[23] * vr),
        1.f / (pQuantizeLUT[24] * vr), 1.f / (pQuantizeLUT[25] * vr), 1.f / (pQuantizeLUT[26] * vr), 1.f / (pQuantizeLUT[27] * vr),
        1.f / (pQuantizeLUT[28] * vr), 1.f / (pQuantizeLUT[29] * vr), 1.f / (pQuantizeLUT[30] * vr), 1.f / (pQuantizeLUT[31] * vr),
        1.f / (pQuantizeLUT[32] * vr), 1.f / (pQuantizeLUT[33] * vr), 1.f / (pQuantizeLUT[34] * vr), 1.f / (pQuantizeLUT[35] * vr),
        1.f / (pQuantizeLUT[36] * vr), 1.f / (pQuantizeLUT[37] * vr), 1.f / (pQuantizeLUT[38] * vr), 1.f / (pQuantizeLUT[39] * vr),
        1.f / (pQuantizeLUT[40] * vr), 1.f / (pQuantizeLUT[41] * vr), 1.f / (pQuantizeLUT[42] * vr), 1.f / (pQuantizeLUT[43] * vr),
        1.f / (pQuantizeLUT[44] * vr), 1.f / (pQuantizeLUT[45] * vr), 1.f / (pQuantizeLUT[46] * vr), 1.f / (pQuantizeLUT[47] * vr),
        1.f / (pQuantizeLUT[48] * vr), 1.f / (pQuantizeLUT[49] * vr), 1.f / (pQuantizeLUT[50] * vr), 1.f / (pQuantizeLUT[51] * vr),
        1.f / (pQuantizeLUT[52] * vr), 1.f / (pQuantizeLUT[53] * vr), 1.f / (pQuantizeLUT[54] * vr), 1.f / (pQuantizeLUT[55] * vr),
        1.f / (pQuantizeLUT[56] * vr), 1.f / (pQuantizeLUT[57] * vr), 1.f / (pQuantizeLUT[58] * vr), 1.f / (pQuantizeLUT[59] * vr),
        1.f / (pQuantizeLUT[60] * vr), 1.f / (pQuantizeLUT[61] * vr), 1.f / (pQuantizeLUT[62] * vr), 1.f / (pQuantizeLUT[63] * vr)
      };

      for (size_t x = 0; x < sizeX; x += 8)
      {
        // Acquire block.
        for (size_t i = 0; i < 8; i++)
          pIntBuffer->u64[i] = *reinterpret_cast<const uint64_t *>(pBlockStart + sizeX * i);

        // Convert to float.
        for (size_t i = 0; i < 64; i++)
          fBuffer[i] = pIntBuffer->u8[i] / 255.f;

        // Swap dimensions.
        for (size_t i = 0; i < 8; i++)
          for (size_t j = i + 1; j < 8; j++)
            std::swap(fBuffer[j + i * 8], fBuffer[i + j * 8]);

        // Apply discrete cosine transform.
        for (size_t i = 0; i < 8; i++)
          inplace_dct8(fBuffer + i * 8);

        // Swap dimensions.
        for (size_t i = 0; i < 8; i++)
          for (size_t j = i + 1; j < 8; j++)
            std::swap(fBuffer[j + i * 8], fBuffer[i + j * 8]);

        // Apply discrete cosine transform.
        for (size_t i = 0; i < 8; i++)
          inplace_dct8(fBuffer + i * 8);

        // Convert & Store.
        for (size_t i = 0; i < 64; i++)
          pTarget[i] = (uint8_t)roundf(_clamp((fBuffer[i] * qTable[i]) + subtract, 0.f, 1.f) * 255.f);

        pTarget += 64;
        pBlockStart += 8;
      }
    }
  };

  internal::intBuffer_t intBuffer;
  float fBuffer[64];

  const uint8_t *pLine = pFrom;

  for (size_t y = 0; y < sizeY / 2; y += 8)
  {
    if (y < startY)
    {
      pLine += 8 * sizeX;
      pTo += 8 * sizeX;

      continue;
    }
    else if (y > endY)
    {
      break;
    }

    const uint8_t *pBlockStart = pLine;
    internal::encode_line(sizeX, &intBuffer, fBuffer, pQuantizeLUT, pBlockStart, pTo);

    pTo += 8 * sizeX;
    pLine += 8 * sizeX;
  }
}

//////////////////////////////////////////////////////////////////////////

inline static float reinterpret_to_float(const int32_t value)
{
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
  return *reinterpret_cast<const float *>(&value);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}

template <size_t index>
inline int32_t _mm_extract_ps_sse2_internal(const __m128 v)
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4324)
#endif
  union _ALIGN(16)
  {
    float _float;
    int32_t _int;
  };
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  _mm_store_ps1(&_float, _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), index * 4)));

  return _int;
}

#define _mm_extract_ps_sse2(v, index) _mm_extract_ps_sse2_internal<index>(v)

inline void _VECTORCALL inplace_dct8_sse2(IN __m128 *pBuffer)
{
  constexpr float C_a = 1.3870398453221474618216191915664f;  // sqrt(2) * cos(1 * pi / 16)
  constexpr float C_b = 1.3065629648763765278566431734272f;  // sqrt(2) * cos(2 * pi / 16)
  constexpr float C_c = 1.1758756024193587169744671046113f;  // sqrt(2) * cos(3 * pi / 16)
  constexpr float C_d = 0.78569495838710218127789736765722f; // sqrt(2) * cos(5 * pi / 16)
  constexpr float C_e = 0.54119610014619698439972320536639f; // sqrt(2) * cos(6 * pi / 16)
  constexpr float C_f = 0.27589937928294301233595756366937f; // sqrt(2) * cos(7 * pi / 16)
  constexpr float C_norm = 0.35355339059327376220042218105242f; // 1 / sqrt(8)    

  // Packing:
  // _1 => pBuffer[0]: (0, 1, 2, 3)
  // _1 => pBuffer[1]:             (4, 5, 6, 7)
  // _2 => pBuffer[2]: (0, 1, 2, 3)
  // _2 => pBuffer[3]:             (4, 5, 6, 7)

  // Step 1:
  // const i_t x07p = pBuffer[0] + pBuffer[7];
  // const i_t x16p = pBuffer[1] + pBuffer[6];
  // const i_t x25p = pBuffer[2] + pBuffer[5];
  // const i_t x34p = pBuffer[3] + pBuffer[4];

  // Prepare:
  const __m128i xp_shuffle_1 = _mm_shuffle_epi32(_mm_castps_si128(pBuffer[1]), _MM_SHUFFLE_REVERSE(3, 2, 1, 0));
  const __m128i xp_shuffle_2 = _mm_shuffle_epi32(_mm_castps_si128(pBuffer[3]), _MM_SHUFFLE_REVERSE(3, 2, 1, 0));

  // Execute:
  const __m128 xp_1 = _mm_add_ps(pBuffer[0], _mm_castsi128_ps(xp_shuffle_1));
  const __m128 xp_2 = _mm_add_ps(pBuffer[2], _mm_castsi128_ps(xp_shuffle_2));

  // Step 2:
  // const i_t x07m = pBuffer[0] - pBuffer[7];  => x07m = pBuffer[0] + -pBuffer[7];
  // const i_t x61m = pBuffer[6] - pBuffer[1];  => x61m = -pBuffer[1] + pBuffer[6];
  // const i_t x25m = pBuffer[2] - pBuffer[5];  => x25m = pBuffer[2] + -pBuffer[5];
  // const i_t x43m = pBuffer[4] - pBuffer[3];  => x43m = -pBuffer[3] + pBuffer[4];

  // Prepare:
  // Previously:
  // const __m128 xprepFlag0 = _mm_set_ps_reverse(1, -1, 1, -1);
  // const __m128 xprep0_1 = _mm_mul_ps(pBuffer[0], xprepFlag0);
  // const __m128 xprep0_2 = _mm_mul_ps(pBuffer[2], xprepFlag0);
  // const __m128 xprep1_1 = _mm_mul_ps(pBuffer[1], xprepFlag0);
  // const __m128 xprep1_2 = _mm_mul_ps(pBuffer[3], xprepFlag0);

  // But this can be simplified to this by flipping the sign bit of the IEEE Single Precision Float Representation rather than multiplying by -1 (xor vs floating point multiply)
  const __m128i xprepFlag0 = _mm_set_epi32_reverse(0, (int32_t)0b10000000000000000000000000000000, 0, (int32_t)0b10000000000000000000000000000000);
  const __m128 xprep0_1 = _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(pBuffer[0]), xprepFlag0));
  const __m128 xprep0_2 = _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(pBuffer[2]), xprepFlag0));
  const __m128 xprep1_1 = _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(pBuffer[1]), xprepFlag0));
  const __m128 xprep1_2 = _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(pBuffer[3]), xprepFlag0));

  const __m128i xprep_shuffle_1 = _mm_shuffle_epi32(_mm_castps_si128(xprep1_1), _MM_SHUFFLE_REVERSE(3, 2, 1, 0));
  const __m128i xprep_shuffle_2 = _mm_shuffle_epi32(_mm_castps_si128(xprep1_2), _MM_SHUFFLE_REVERSE(3, 2, 1, 0));

  // Execute:
  const __m128 xm_1 = _mm_add_ps(xprep0_1, _mm_castsi128_ps(xprep_shuffle_1));
  const __m128 xm_2 = _mm_add_ps(xprep0_2, _mm_castsi128_ps(xprep_shuffle_2));

  // Step 3:
  // const i_t x07p34pp = x07p + x34p;
  // const i_t x07p34pm = x07p - x34p;
  // const i_t x16p25pp = x16p + x25p;
  // const i_t x16p25pm = x16p - x25p;

  // Prepare:
  // Pack _1 and _2 into one __m128i.
  const __m128i xpp_prep_front_flag = _mm_set_epi32_reverse(-1, -1, 0, 0);
  const __m128i xpp_prep_back_flag = _mm_set_epi32_reverse(0, 0, -1, -1);
  const __m128i xpp_prep0_12 = _mm_or_si128(_mm_and_si128(_mm_castps_si128(xp_1), xpp_prep_front_flag), _mm_slli_si128(_mm_castps_si128(xp_2), 8)); // (x07p_1, x16p_1, x07p_2, x16p_2)
  const __m128i xpp_prep1_12 = _mm_or_si128(_mm_srli_si128(_mm_castps_si128(xp_1), 8), _mm_and_si128(_mm_castps_si128(xp_2), xpp_prep_back_flag)); // (x25p_1, x34p_1, x25p_2, x34p_2)

  // We need to shuffle to swap the x25 s and x34 s.
  const __m128i xpp_prep2_12 = _mm_shuffle_epi32(xpp_prep1_12, _MM_SHUFFLE_REVERSE(1, 0, 3, 2)); // (x34p_1, x25p_1, x34p_2, x25p_2)

  // Execute:
  const __m128 xppp_12 = _mm_add_ps(_mm_castsi128_ps(xpp_prep0_12), _mm_castsi128_ps(xpp_prep2_12)); // (x07p34pp_1, x16p25pp_1, x07p34pp_2, x16p25pp_2)
  const __m128 xppm_12 = _mm_sub_ps(_mm_castsi128_ps(xpp_prep0_12), _mm_castsi128_ps(xpp_prep2_12)); // (x07p34pm_1, x16p25pm_1, x07p34pm_2, x16p25pm_2)

  // Step 4:
  // pBuffer[0] = (C_norm * (x07p34pp + x16p25pp));
  // pBuffer[1] = (C_norm * (((C_a * x07m)) - ((C_c * x61m)) + ((C_d * x25m)) - ((C_f * x43m))));
  // pBuffer[2] = (C_norm * (((C_b * x07p34pm)) + ((C_e * x16p25pm))));
  // pBuffer[3] = (C_norm * (((C_c * x07m)) + ((C_f * x61m)) - ((C_a * x25m)) + ((C_d * x43m))));
  // 
  // pBuffer[4] = (C_norm * (x07p34pp - x16p25pp));
  // pBuffer[5] = (C_norm * (((C_d * x07m)) + ((C_a * x61m)) + ((C_f * x25m)) - ((C_c * x43m))));
  // pBuffer[6] = (C_norm * (((C_e * x07p34pm)) - ((C_b * x16p25pm))));
  // pBuffer[7] = (C_norm * (((C_f * x07m)) + ((C_d * x61m)) + ((C_c * x25m)) + ((C_a * x43m))));

  // We will break this down into multiple steps:

  // Step 4.1:
  // const i_t xf11 = (C_a * x07m);
  // const i_t xf31 = (C_c * x07m);
  // const i_t xf51 = (C_d * x07m);
  // const i_t xf71 = (C_f * x07m);
  // 
  // const i_t xf12 = (-C_c * x16m);
  // const i_t xf32 = (C_f * x16m);
  // const i_t xf52 = (C_a * x16m);
  // const i_t xf72 = (C_d * x16m);
  // 
  // const i_t xf13 = (C_d * x25m);
  // const i_t xf33 = (-C_a * x25m);
  // const i_t xf53 = (C_f * x25m);
  // const i_t xf73 = (C_c * x25m);
  // 
  // const i_t xf14 = (C_f * x34m);
  // const i_t xf34 = (C_d * x34m);
  // const i_t xf54 = (-C_c * x34m);
  // const i_t xf74 = (C_a * x34m);

  // Prepare:
  const __m128 xf_1_factors = _mm_set_ps_reverse(C_a, C_c, C_d, C_f);
  const __m128 xf_3_factors = _mm_set_ps_reverse(-C_c, C_f, C_a, C_d);
  const __m128 xf_5_factors = _mm_set_ps_reverse(C_d, -C_a, C_f, C_c);
  const __m128 xf_7_factors = _mm_set_ps_reverse(C_f, C_d, -C_c, C_a);

  const __m128 x07m_1 = _mm_set1_ps(reinterpret_to_float(_mm_extract_ps_sse2(xm_1, 0)));
  const __m128 x07m_2 = _mm_set1_ps(reinterpret_to_float(_mm_extract_ps_sse2(xm_2, 0)));
  const __m128 x16m_1 = _mm_set1_ps(reinterpret_to_float(_mm_extract_ps_sse2(xm_1, 1)));
  const __m128 x16m_2 = _mm_set1_ps(reinterpret_to_float(_mm_extract_ps_sse2(xm_2, 1)));
  const __m128 x25m_1 = _mm_set1_ps(reinterpret_to_float(_mm_extract_ps_sse2(xm_1, 2)));
  const __m128 x25m_2 = _mm_set1_ps(reinterpret_to_float(_mm_extract_ps_sse2(xm_2, 2)));
  const __m128 x34m_1 = _mm_set1_ps(reinterpret_to_float(_mm_extract_ps_sse2(xm_1, 3)));
  const __m128 x34m_2 = _mm_set1_ps(reinterpret_to_float(_mm_extract_ps_sse2(xm_2, 3)));

  // Execute:
  const __m128 xf_1_1 = _mm_mul_ps(x07m_1, xf_1_factors);
  const __m128 xf_1_2 = _mm_mul_ps(x07m_2, xf_1_factors);
  const __m128 xf_3_1 = _mm_mul_ps(x16m_1, xf_3_factors);
  const __m128 xf_3_2 = _mm_mul_ps(x16m_2, xf_3_factors);
  const __m128 xf_5_1 = _mm_mul_ps(x25m_1, xf_5_factors);
  const __m128 xf_5_2 = _mm_mul_ps(x25m_2, xf_5_factors);
  const __m128 xf_7_1 = _mm_mul_ps(x34m_1, xf_7_factors);
  const __m128 xf_7_2 = _mm_mul_ps(x34m_2, xf_7_factors);

  // Step 4.2:
  // const i_t xff1 = xf11 + xf12 + xf13 + xf14;
  // const i_t xff3 = xf31 + xf32 + xf33 + xf34;
  // const i_t xff5 = xf51 + xf52 + xf53 + xf54;
  // const i_t xff7 = xf71 + xf72 + xf73 + xf74;

  const __m128 xff1357_1 = _mm_add_ps(_mm_add_ps(xf_1_1, xf_3_1), _mm_add_ps(xf_5_1, xf_7_1)); // (xff1_1, xff3_1, xff5_1, xff7_1)
  const __m128 xff1357_2 = _mm_add_ps(_mm_add_ps(xf_1_2, xf_3_2), _mm_add_ps(xf_5_2, xf_7_2)); // (xff1_2, xff3_2, xff5_2, xff7_2)

  // Step 4.3:
  // const i_t xff0 = x07p34pp + x16p25pp;
  // const i_t xff4 = x07p34pp - x16p25pp;

  // Prepare:
  const __m128i xpppshift1_12 = _mm_srli_si128(_mm_castps_si128(xppp_12), 4); // Shift the values over by one float: (x16p25pp_1, x07p34pp_2, x16p25pp_2, ZERO)

  // Execute:
  const __m128 xff0_12 = _mm_add_ps(xppp_12, _mm_castsi128_ps(xpppshift1_12)); // (xf0_1, ---, xf0_2, ---)
  const __m128 xff4_12 = _mm_sub_ps(xppp_12, _mm_castsi128_ps(xpppshift1_12)); // (xf4_1, ---, xf4_2, ---)

  // Step 4.4:
  // const i_t xff2s1 = (C_b * x07p34pm);
  // const i_t xff2s2 = (C_e * x16p25pm);
  // const i_t xff6s1 = (C_e * x07p34pm);
  // const i_t xff6s2 = (-C_b * x16p25pm);

  // Prepare:
  const __m128 xff2_prep = _mm_set_ps_reverse(C_b, C_e, C_b, C_e);
  const __m128 xff6_prep = _mm_set_ps_reverse(C_e, -C_b, C_e, -C_b);

  // Execute:
  const __m128 xff2s_12 = _mm_mul_ps(xppm_12, xff2_prep); // (xff2s1_1, xff2s2_1, xff2s1_2, xff2s2_2)
  const __m128 xff6s_12 = _mm_mul_ps(xppm_12, xff6_prep); // (xff6s1_1, xff6s2_1, xff6s1_2, xff6s2_2)

  // Step 4.5:
  // const i_t xff2 = xff2s1 + xff2s2;
  // const i_t xff6 = xff6s1 + xff6s2;

  // Prepare:
  const __m128i xff2shift1_12 = _mm_srli_si128(_mm_castps_si128(xff2s_12), 4); // Shift the values over by one float: (xff2s2_1, xff2s1_2, xff2s2_2, ZERO)
  const __m128i xff6shift1_12 = _mm_srli_si128(_mm_castps_si128(xff6s_12), 4); // Shift the values over by one float: (xff6s2_1, xff6s1_2, xff6s2_2, ZERO)

  // Execute:
  const __m128 xff2_12 = _mm_add_ps(xff2s_12, _mm_castsi128_ps(xff2shift1_12)); // (xff2_1, ---, xff2_2, ---)
  const __m128 xff6_12 = _mm_add_ps(xff6s_12, _mm_castsi128_ps(xff6shift1_12)); // (xff6_1, ---, xff6_2, ---)

  // Step 4.6: Pack values

  // Prepare:
  const __m128i flag0 = _mm_set_epi32_reverse(-1, 0, 0, 0);
  const __m128i flag2 = _mm_set_epi32_reverse(0, 0, -1, 0);
  const __m128i flag13 = _mm_set_epi32_reverse(0, -1, 0, -1);

  // Execute:
  __m128i buffer0 = _mm_or_si128(_mm_and_si128(flag0, _mm_castps_si128(xff0_12)), _mm_and_si128(flag2, _mm_slli_si128(_mm_castps_si128(xff2_12), 2 * sizeof(float)))); // (xff0_1, ZERO, xff2_1, ZERO)
  __m128i buffer1 = _mm_or_si128(_mm_and_si128(flag0, _mm_castps_si128(xff4_12)), _mm_and_si128(flag2, _mm_slli_si128(_mm_castps_si128(xff6_12), 2 * sizeof(float)))); // (xff4_1, ZERO, xff6_1, ZERO)
  __m128i buffer2 = _mm_or_si128(_mm_and_si128(flag0, _mm_srli_si128(_mm_castps_si128(xff0_12), 2 * sizeof(float))), _mm_and_si128(flag2, _mm_castps_si128(xff2_12))); // (xff0_2, ZERO, xff2_2, ZERO)
  __m128i buffer3 = _mm_or_si128(_mm_and_si128(flag0, _mm_srli_si128(_mm_castps_si128(xff4_12), 2 * sizeof(float))), _mm_and_si128(flag2, _mm_castps_si128(xff6_12))); // (xff4_2, ZERO, xff6_2, ZERO)

  buffer0 = _mm_or_si128(buffer0, _mm_and_si128(flag13, _mm_shuffle_epi32(_mm_castps_si128(xff1357_1), _MM_SHUFFLE_REVERSE(2, 0, 3, 1)))); // (xff0_1, xff1_1, xff2_1, xff3_1)
  buffer1 = _mm_or_si128(buffer1, _mm_and_si128(flag13, _mm_shuffle_epi32(_mm_castps_si128(xff1357_1), _MM_SHUFFLE_REVERSE(0, 2, 1, 3)))); // (xff4_1, xff5_1, xff6_1, xff7_1)
  buffer2 = _mm_or_si128(buffer2, _mm_and_si128(flag13, _mm_shuffle_epi32(_mm_castps_si128(xff1357_2), _MM_SHUFFLE_REVERSE(2, 0, 3, 1)))); // (xff0_2, xff1_2, xff2_2, xff3_2)
  buffer3 = _mm_or_si128(buffer3, _mm_and_si128(flag13, _mm_shuffle_epi32(_mm_castps_si128(xff1357_2), _MM_SHUFFLE_REVERSE(0, 2, 1, 3)))); // (xff4_2, xff5_2, xff6_2, xff7_2)

  // Step 4.7:
  // pBuffer[0] = (C_norm * xff0);
  // pBuffer[1] = (C_norm * xff1);
  // pBuffer[2] = (C_norm * xff2);
  // pBuffer[3] = (C_norm * xff3);
  //
  // pBuffer[4] = (C_norm * xff4);
  // pBuffer[5] = (C_norm * xff5);
  // pBuffer[6] = (C_norm * xff6);
  // pBuffer[7] = (C_norm * xff7);

  // Prepare:
  const __m128 cnorm = _mm_set1_ps(C_norm);

  // Execute:
  pBuffer[0] = _mm_mul_ps(cnorm, _mm_castsi128_ps(buffer0));
  pBuffer[1] = _mm_mul_ps(cnorm, _mm_castsi128_ps(buffer1));
  pBuffer[2] = _mm_mul_ps(cnorm, _mm_castsi128_ps(buffer2));
  pBuffer[3] = _mm_mul_ps(cnorm, _mm_castsi128_ps(buffer3));
}

// The SSE4.1 Variant is the most compliant and fastest variant.
// Since the required components of SSE4.1 are either very minor (`_mm_extract_ps` for extracting a specifics float from a __m128) 
//  or can very easily be implemented without SSE4.1 (`_mm_cvtepu8_epi32` for expanding 8 bit unsigned integers to 32 bit signed integers),
//  specific derivates were made for SSSE3 (nearly as fast as SSE4.1) and SSE2 (noticeably slower, but still way faster than the naive implementation).
#ifndef _MSC_VER
__attribute__((target("sse4.1")))
#endif
inline static void _VECTORCALL inplace_dct8_sse41(IN __m128 *pBuffer)
{
  constexpr float C_a = 1.3870398453221474618216191915664f;  // sqrt(2) * cos(1 * pi / 16)
  constexpr float C_b = 1.3065629648763765278566431734272f;  // sqrt(2) * cos(2 * pi / 16)
  constexpr float C_c = 1.1758756024193587169744671046113f;  // sqrt(2) * cos(3 * pi / 16)
  constexpr float C_d = 0.78569495838710218127789736765722f; // sqrt(2) * cos(5 * pi / 16)
  constexpr float C_e = 0.54119610014619698439972320536639f; // sqrt(2) * cos(6 * pi / 16)
  constexpr float C_f = 0.27589937928294301233595756366937f; // sqrt(2) * cos(7 * pi / 16)
  constexpr float C_norm = 0.35355339059327376220042218105242f; // 1 / sqrt(8)    

  // Packing:
  // _1 => pBuffer[0]: (0, 1, 2, 3)
  // _1 => pBuffer[1]:             (4, 5, 6, 7)
  // _2 => pBuffer[2]: (0, 1, 2, 3)
  // _2 => pBuffer[3]:             (4, 5, 6, 7)

  // Step 1:
  // const i_t x07p = pBuffer[0] + pBuffer[7];
  // const i_t x16p = pBuffer[1] + pBuffer[6];
  // const i_t x25p = pBuffer[2] + pBuffer[5];
  // const i_t x34p = pBuffer[3] + pBuffer[4];

  // Prepare:
  const __m128i xp_shuffle_1 = _mm_shuffle_epi32(_mm_castps_si128(pBuffer[1]), _MM_SHUFFLE_REVERSE(3, 2, 1, 0));
  const __m128i xp_shuffle_2 = _mm_shuffle_epi32(_mm_castps_si128(pBuffer[3]), _MM_SHUFFLE_REVERSE(3, 2, 1, 0));

  // Execute:
  const __m128 xp_1 = _mm_add_ps(pBuffer[0], _mm_castsi128_ps(xp_shuffle_1));
  const __m128 xp_2 = _mm_add_ps(pBuffer[2], _mm_castsi128_ps(xp_shuffle_2));

  // Step 2:
  // const i_t x07m = pBuffer[0] - pBuffer[7];  => x07m = pBuffer[0] + -pBuffer[7];
  // const i_t x61m = pBuffer[6] - pBuffer[1];  => x61m = -pBuffer[1] + pBuffer[6];
  // const i_t x25m = pBuffer[2] - pBuffer[5];  => x25m = pBuffer[2] + -pBuffer[5];
  // const i_t x43m = pBuffer[4] - pBuffer[3];  => x43m = -pBuffer[3] + pBuffer[4];

  // Prepare:
  // Previously:
  // const __m128 xprepFlag0 = _mm_set_ps_reverse(1, -1, 1, -1);
  // const __m128 xprep0_1 = _mm_mul_ps(pBuffer[0], xprepFlag0);
  // const __m128 xprep0_2 = _mm_mul_ps(pBuffer[2], xprepFlag0);
  // const __m128 xprep1_1 = _mm_mul_ps(pBuffer[1], xprepFlag0);
  // const __m128 xprep1_2 = _mm_mul_ps(pBuffer[3], xprepFlag0);

  // But this can be simplified to this by flipping the sign bit of the IEEE Single Precision Float Representation rather than multiplying by -1 (xor vs floating point multiply)
  const __m128i xprepFlag0 = _mm_set_epi32_reverse(0, (int32_t)0b10000000000000000000000000000000, 0, (int32_t)0b10000000000000000000000000000000);
  const __m128 xprep0_1 = _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(pBuffer[0]), xprepFlag0));
  const __m128 xprep0_2 = _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(pBuffer[2]), xprepFlag0));
  const __m128 xprep1_1 = _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(pBuffer[1]), xprepFlag0));
  const __m128 xprep1_2 = _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(pBuffer[3]), xprepFlag0));

  const __m128i xprep_shuffle_1 = _mm_shuffle_epi32(_mm_castps_si128(xprep1_1), _MM_SHUFFLE_REVERSE(3, 2, 1, 0));
  const __m128i xprep_shuffle_2 = _mm_shuffle_epi32(_mm_castps_si128(xprep1_2), _MM_SHUFFLE_REVERSE(3, 2, 1, 0));

  // Execute:
  const __m128 xm_1 = _mm_add_ps(xprep0_1, _mm_castsi128_ps(xprep_shuffle_1));
  const __m128 xm_2 = _mm_add_ps(xprep0_2, _mm_castsi128_ps(xprep_shuffle_2));

  // Step 3:
  // const i_t x07p34pp = x07p + x34p;
  // const i_t x07p34pm = x07p - x34p;
  // const i_t x16p25pp = x16p + x25p;
  // const i_t x16p25pm = x16p - x25p;

  // Prepare:
  // Pack _1 and _2 into one __m128i.
  const __m128i xpp_prep_front_flag = _mm_set_epi32_reverse(-1, -1, 0, 0);
  const __m128i xpp_prep_back_flag = _mm_set_epi32_reverse(0, 0, -1, -1);
  const __m128i xpp_prep0_12 = _mm_or_si128(_mm_and_si128(_mm_castps_si128(xp_1), xpp_prep_front_flag), _mm_slli_si128(_mm_castps_si128(xp_2), 8)); // (x07p_1, x16p_1, x07p_2, x16p_2)
  const __m128i xpp_prep1_12 = _mm_or_si128(_mm_srli_si128(_mm_castps_si128(xp_1), 8), _mm_and_si128(_mm_castps_si128(xp_2), xpp_prep_back_flag)); // (x25p_1, x34p_1, x25p_2, x34p_2)

  // We need to shuffle to swap the x25 s and x34 s.
  const __m128i xpp_prep2_12 = _mm_shuffle_epi32(xpp_prep1_12, _MM_SHUFFLE_REVERSE(1, 0, 3, 2)); // (x34p_1, x25p_1, x34p_2, x25p_2)

  // Execute:
  const __m128 xppp_12 = _mm_add_ps(_mm_castsi128_ps(xpp_prep0_12), _mm_castsi128_ps(xpp_prep2_12)); // (x07p34pp_1, x16p25pp_1, x07p34pp_2, x16p25pp_2)
  const __m128 xppm_12 = _mm_sub_ps(_mm_castsi128_ps(xpp_prep0_12), _mm_castsi128_ps(xpp_prep2_12)); // (x07p34pm_1, x16p25pm_1, x07p34pm_2, x16p25pm_2)

  // Step 4:
  // pBuffer[0] = (C_norm * (x07p34pp + x16p25pp));
  // pBuffer[1] = (C_norm * (((C_a * x07m)) - ((C_c * x61m)) + ((C_d * x25m)) - ((C_f * x43m))));
  // pBuffer[2] = (C_norm * (((C_b * x07p34pm)) + ((C_e * x16p25pm))));
  // pBuffer[3] = (C_norm * (((C_c * x07m)) + ((C_f * x61m)) - ((C_a * x25m)) + ((C_d * x43m))));
  // 
  // pBuffer[4] = (C_norm * (x07p34pp - x16p25pp));
  // pBuffer[5] = (C_norm * (((C_d * x07m)) + ((C_a * x61m)) + ((C_f * x25m)) - ((C_c * x43m))));
  // pBuffer[6] = (C_norm * (((C_e * x07p34pm)) - ((C_b * x16p25pm))));
  // pBuffer[7] = (C_norm * (((C_f * x07m)) + ((C_d * x61m)) + ((C_c * x25m)) + ((C_a * x43m))));

  // We will break this down into multiple steps:

  // Step 4.1:
  // const i_t xf11 = (C_a * x07m);
  // const i_t xf31 = (C_c * x07m);
  // const i_t xf51 = (C_d * x07m);
  // const i_t xf71 = (C_f * x07m);
  // 
  // const i_t xf12 = (-C_c * x16m);
  // const i_t xf32 = (C_f * x16m);
  // const i_t xf52 = (C_a * x16m);
  // const i_t xf72 = (C_d * x16m);
  // 
  // const i_t xf13 = (C_d * x25m);
  // const i_t xf33 = (-C_a * x25m);
  // const i_t xf53 = (C_f * x25m);
  // const i_t xf73 = (C_c * x25m);
  // 
  // const i_t xf14 = (C_f * x34m);
  // const i_t xf34 = (C_d * x34m);
  // const i_t xf54 = (-C_c * x34m);
  // const i_t xf74 = (C_a * x34m);

  // Prepare:
  const __m128 xf_1_factors = _mm_set_ps_reverse(C_a, C_c, C_d, C_f);
  const __m128 xf_3_factors = _mm_set_ps_reverse(-C_c, C_f, C_a, C_d);
  const __m128 xf_5_factors = _mm_set_ps_reverse(C_d, -C_a, C_f, C_c);
  const __m128 xf_7_factors = _mm_set_ps_reverse(C_f, C_d, -C_c, C_a);

  const __m128 x07m_1 = _mm_set1_ps(reinterpret_to_float(_mm_extract_ps(xm_1, 0))); // _mm_extract_ps is SSE4.1 (!) 
  const __m128 x07m_2 = _mm_set1_ps(reinterpret_to_float(_mm_extract_ps(xm_2, 0)));
  const __m128 x16m_1 = _mm_set1_ps(reinterpret_to_float(_mm_extract_ps(xm_1, 1)));
  const __m128 x16m_2 = _mm_set1_ps(reinterpret_to_float(_mm_extract_ps(xm_2, 1)));
  const __m128 x25m_1 = _mm_set1_ps(reinterpret_to_float(_mm_extract_ps(xm_1, 2)));
  const __m128 x25m_2 = _mm_set1_ps(reinterpret_to_float(_mm_extract_ps(xm_2, 2)));
  const __m128 x34m_1 = _mm_set1_ps(reinterpret_to_float(_mm_extract_ps(xm_1, 3)));
  const __m128 x34m_2 = _mm_set1_ps(reinterpret_to_float(_mm_extract_ps(xm_2, 3)));

  // Execute:
  const __m128 xf_1_1 = _mm_mul_ps(x07m_1, xf_1_factors);
  const __m128 xf_1_2 = _mm_mul_ps(x07m_2, xf_1_factors);
  const __m128 xf_3_1 = _mm_mul_ps(x16m_1, xf_3_factors);
  const __m128 xf_3_2 = _mm_mul_ps(x16m_2, xf_3_factors);
  const __m128 xf_5_1 = _mm_mul_ps(x25m_1, xf_5_factors);
  const __m128 xf_5_2 = _mm_mul_ps(x25m_2, xf_5_factors);
  const __m128 xf_7_1 = _mm_mul_ps(x34m_1, xf_7_factors);
  const __m128 xf_7_2 = _mm_mul_ps(x34m_2, xf_7_factors);

  // Step 4.2:
  // const i_t xff1 = xf11 + xf12 + xf13 + xf14;
  // const i_t xff3 = xf31 + xf32 + xf33 + xf34;
  // const i_t xff5 = xf51 + xf52 + xf53 + xf54;
  // const i_t xff7 = xf71 + xf72 + xf73 + xf74;

  const __m128 xff1357_1 = _mm_add_ps(_mm_add_ps(xf_1_1, xf_3_1), _mm_add_ps(xf_5_1, xf_7_1)); // (xff1_1, xff3_1, xff5_1, xff7_1)
  const __m128 xff1357_2 = _mm_add_ps(_mm_add_ps(xf_1_2, xf_3_2), _mm_add_ps(xf_5_2, xf_7_2)); // (xff1_2, xff3_2, xff5_2, xff7_2)

  // Step 4.3:
  // const i_t xff0 = x07p34pp + x16p25pp;
  // const i_t xff4 = x07p34pp - x16p25pp;

  // Prepare:
  const __m128i xpppshift1_12 = _mm_srli_si128(_mm_castps_si128(xppp_12), 4); // Shift the values over by one float: (x16p25pp_1, x07p34pp_2, x16p25pp_2, ZERO)

  // Execute:
  const __m128 xff0_12 = _mm_add_ps(xppp_12, _mm_castsi128_ps(xpppshift1_12)); // (xf0_1, ---, xf0_2, ---)
  const __m128 xff4_12 = _mm_sub_ps(xppp_12, _mm_castsi128_ps(xpppshift1_12)); // (xf4_1, ---, xf4_2, ---)

  // Step 4.4:
  // const i_t xff2s1 = (C_b * x07p34pm);
  // const i_t xff2s2 = (C_e * x16p25pm);
  // const i_t xff6s1 = (C_e * x07p34pm);
  // const i_t xff6s2 = (-C_b * x16p25pm);

  // Prepare:
  const __m128 xff2_prep = _mm_set_ps_reverse(C_b, C_e, C_b, C_e);
  const __m128 xff6_prep = _mm_set_ps_reverse(C_e, -C_b, C_e, -C_b);

  // Execute:
  const __m128 xff2s_12 = _mm_mul_ps(xppm_12, xff2_prep); // (xff2s1_1, xff2s2_1, xff2s1_2, xff2s2_2)
  const __m128 xff6s_12 = _mm_mul_ps(xppm_12, xff6_prep); // (xff6s1_1, xff6s2_1, xff6s1_2, xff6s2_2)

  // Step 4.5:
  // const i_t xff2 = xff2s1 + xff2s2;
  // const i_t xff6 = xff6s1 + xff6s2;

  // Prepare:
  const __m128i xff2shift1_12 = _mm_srli_si128(_mm_castps_si128(xff2s_12), 4); // Shift the values over by one float: (xff2s2_1, xff2s1_2, xff2s2_2, ZERO)
  const __m128i xff6shift1_12 = _mm_srli_si128(_mm_castps_si128(xff6s_12), 4); // Shift the values over by one float: (xff6s2_1, xff6s1_2, xff6s2_2, ZERO)

  // Execute:
  const __m128 xff2_12 = _mm_add_ps(xff2s_12, _mm_castsi128_ps(xff2shift1_12)); // (xff2_1, ---, xff2_2, ---)
  const __m128 xff6_12 = _mm_add_ps(xff6s_12, _mm_castsi128_ps(xff6shift1_12)); // (xff6_1, ---, xff6_2, ---)

  // Step 4.6: Pack values

  // Prepare:
  const __m128i flag0 = _mm_set_epi32_reverse(-1, 0, 0, 0);
  const __m128i flag2 = _mm_set_epi32_reverse(0, 0, -1, 0);
  const __m128i flag13 = _mm_set_epi32_reverse(0, -1, 0, -1);

  // Execute:
  __m128i buffer0 = _mm_or_si128(_mm_and_si128(flag0, _mm_castps_si128(xff0_12)), _mm_and_si128(flag2, _mm_slli_si128(_mm_castps_si128(xff2_12), 2 * sizeof(float)))); // (xff0_1, ZERO, xff2_1, ZERO)
  __m128i buffer1 = _mm_or_si128(_mm_and_si128(flag0, _mm_castps_si128(xff4_12)), _mm_and_si128(flag2, _mm_slli_si128(_mm_castps_si128(xff6_12), 2 * sizeof(float)))); // (xff4_1, ZERO, xff6_1, ZERO)
  __m128i buffer2 = _mm_or_si128(_mm_and_si128(flag0, _mm_srli_si128(_mm_castps_si128(xff0_12), 2 * sizeof(float))), _mm_and_si128(flag2, _mm_castps_si128(xff2_12))); // (xff0_2, ZERO, xff2_2, ZERO)
  __m128i buffer3 = _mm_or_si128(_mm_and_si128(flag0, _mm_srli_si128(_mm_castps_si128(xff4_12), 2 * sizeof(float))), _mm_and_si128(flag2, _mm_castps_si128(xff6_12))); // (xff4_2, ZERO, xff6_2, ZERO)

  buffer0 = _mm_or_si128(buffer0, _mm_and_si128(flag13, _mm_shuffle_epi32(_mm_castps_si128(xff1357_1), _MM_SHUFFLE_REVERSE(2, 0, 3, 1)))); // (xff0_1, xff1_1, xff2_1, xff3_1)
  buffer1 = _mm_or_si128(buffer1, _mm_and_si128(flag13, _mm_shuffle_epi32(_mm_castps_si128(xff1357_1), _MM_SHUFFLE_REVERSE(0, 2, 1, 3)))); // (xff4_1, xff5_1, xff6_1, xff7_1)
  buffer2 = _mm_or_si128(buffer2, _mm_and_si128(flag13, _mm_shuffle_epi32(_mm_castps_si128(xff1357_2), _MM_SHUFFLE_REVERSE(2, 0, 3, 1)))); // (xff0_2, xff1_2, xff2_2, xff3_2)
  buffer3 = _mm_or_si128(buffer3, _mm_and_si128(flag13, _mm_shuffle_epi32(_mm_castps_si128(xff1357_2), _MM_SHUFFLE_REVERSE(0, 2, 1, 3)))); // (xff4_2, xff5_2, xff6_2, xff7_2)

  // Step 4.7:
  // pBuffer[0] = (C_norm * xff0);
  // pBuffer[1] = (C_norm * xff1);
  // pBuffer[2] = (C_norm * xff2);
  // pBuffer[3] = (C_norm * xff3);
  //
  // pBuffer[4] = (C_norm * xff4);
  // pBuffer[5] = (C_norm * xff5);
  // pBuffer[6] = (C_norm * xff6);
  // pBuffer[7] = (C_norm * xff7);

  // Prepare:
  const __m128 cnorm = _mm_set1_ps(C_norm);

  // Execute:
  pBuffer[0] = _mm_mul_ps(cnorm, _mm_castsi128_ps(buffer0));
  pBuffer[1] = _mm_mul_ps(cnorm, _mm_castsi128_ps(buffer1));
  pBuffer[2] = _mm_mul_ps(cnorm, _mm_castsi128_ps(buffer2));
  pBuffer[3] = _mm_mul_ps(cnorm, _mm_castsi128_ps(buffer3));
}

//////////////////////////////////////////////////////////////////////////

void simdDCT_EncodeQuantizeReorderStereoBuffer_SSE41_Float(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY)
{
  struct internal
  {
#ifndef _MSC_VER
    __attribute__((target("sse4.1")))
#endif
      static inline void encode_line(const size_t sizeX, IN const float *pQuantizeLUT, IN const uint8_t *pBlockStart, IN_OUT uint16_t *pOutPositions[64])
    {
      constexpr float vr = .95f;
      constexpr float subtract = 127.0f;

      _ALIGN(16) __m128 qTable[64 / 4];

      qTable[0] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[0] * vr)), (255.0f / (pQuantizeLUT[1] * vr)), (255.0f / (pQuantizeLUT[2] * vr)), (255.0f / (pQuantizeLUT[3] * vr)));
      qTable[1] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[4] * vr)), (255.0f / (pQuantizeLUT[5] * vr)), (255.0f / (pQuantizeLUT[6] * vr)), (255.0f / (pQuantizeLUT[7] * vr)));
      qTable[2] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[8] * vr)), (255.0f / (pQuantizeLUT[9] * vr)), (255.0f / (pQuantizeLUT[10] * vr)), (255.0f / (pQuantizeLUT[11] * vr)));
      qTable[3] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[12] * vr)), (255.0f / (pQuantizeLUT[13] * vr)), (255.0f / (pQuantizeLUT[14] * vr)), (255.0f / (pQuantizeLUT[15] * vr)));
      qTable[4] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[16] * vr)), (255.0f / (pQuantizeLUT[17] * vr)), (255.0f / (pQuantizeLUT[18] * vr)), (255.0f / (pQuantizeLUT[19] * vr)));
      qTable[5] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[20] * vr)), (255.0f / (pQuantizeLUT[21] * vr)), (255.0f / (pQuantizeLUT[22] * vr)), (255.0f / (pQuantizeLUT[23] * vr)));
      qTable[6] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[24] * vr)), (255.0f / (pQuantizeLUT[25] * vr)), (255.0f / (pQuantizeLUT[26] * vr)), (255.0f / (pQuantizeLUT[27] * vr)));
      qTable[7] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[28] * vr)), (255.0f / (pQuantizeLUT[29] * vr)), (255.0f / (pQuantizeLUT[30] * vr)), (255.0f / (pQuantizeLUT[31] * vr)));
      qTable[8] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[32] * vr)), (255.0f / (pQuantizeLUT[33] * vr)), (255.0f / (pQuantizeLUT[34] * vr)), (255.0f / (pQuantizeLUT[35] * vr)));
      qTable[9] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[36] * vr)), (255.0f / (pQuantizeLUT[37] * vr)), (255.0f / (pQuantizeLUT[38] * vr)), (255.0f / (pQuantizeLUT[39] * vr)));
      qTable[10] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[40] * vr)), (255.0f / (pQuantizeLUT[41] * vr)), (255.0f / (pQuantizeLUT[42] * vr)), (255.0f / (pQuantizeLUT[43] * vr)));
      qTable[11] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[44] * vr)), (255.0f / (pQuantizeLUT[45] * vr)), (255.0f / (pQuantizeLUT[46] * vr)), (255.0f / (pQuantizeLUT[47] * vr)));
      qTable[12] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[48] * vr)), (255.0f / (pQuantizeLUT[49] * vr)), (255.0f / (pQuantizeLUT[50] * vr)), (255.0f / (pQuantizeLUT[51] * vr)));
      qTable[13] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[52] * vr)), (255.0f / (pQuantizeLUT[53] * vr)), (255.0f / (pQuantizeLUT[54] * vr)), (255.0f / (pQuantizeLUT[55] * vr)));
      qTable[14] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[56] * vr)), (255.0f / (pQuantizeLUT[57] * vr)), (255.0f / (pQuantizeLUT[58] * vr)), (255.0f / (pQuantizeLUT[59] * vr)));
      qTable[15] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[60] * vr)), (255.0f / (pQuantizeLUT[61] * vr)), (255.0f / (pQuantizeLUT[62] * vr)), (255.0f / (pQuantizeLUT[63] * vr)));

      _ALIGN(16) __m128i localBuffer[8];
      static_assert(sizeof(localBuffer) == sizeof(uint8_t) * 128, "Invalid Size.");

      union _ALIGN(16)
      {
        __m128 intermediateBuffer[128 / sizeof(__m128i) * sizeof(float)];
        float intermediateBufferF[128];
      };

      static_assert(sizeof(intermediateBuffer) == sizeof(float) * 128, "Invalid Packing");
      static_assert(sizeof(intermediateBufferF) == sizeof(__m128i) * 32, "Invalid Packing");

      for (size_t x = 0; x < sizeX; x += 16)
      {
        // Acquire block.
        {
          for (size_t i = 0; i < 8; i++)
            localBuffer[i] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pBlockStart + sizeX * i));
        }

        // Convert to intermediateBuffer.
        {
          const __m128 inverse_0xFF_float = _mm_set1_ps(1.f / (float)0xFF);

          for (size_t i = 0; i < 8; i++)
          {
            intermediateBuffer[i * 4 + 0] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(localBuffer[i]))); // _mm_cvtepu8_epi32 is SSE4.1 (!)
            intermediateBuffer[i * 4 + 1] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(localBuffer[i], 4))));
            intermediateBuffer[i * 4 + 2] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(localBuffer[i], 8))));
            intermediateBuffer[i * 4 + 3] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(localBuffer[i], 12))));
          }
        }

        // Swap dimensions.
        {
          for (size_t i = 0; i < 8; i++)
          {
            for (size_t j = i + 1; j < 8; j++)
            {
              {
                const float tmp = intermediateBufferF[i * 16 + j];
                intermediateBufferF[i * 16 + j] = intermediateBufferF[j * 16 + i];
                intermediateBufferF[j * 16 + i] = tmp;
              }

              {
                const float tmp = intermediateBufferF[i * 16 + j + 8];
                intermediateBufferF[i * 16 + j + 8] = intermediateBufferF[j * 16 + i + 8];
                intermediateBufferF[j * 16 + i + 8] = tmp;
              }
            }
          }
        }

        // Apply discrete cosine transform.
        for (size_t i = 0; i < 8; i++)
          inplace_dct8_sse41(intermediateBuffer + i * 4);

        // Swap dimensions.
        {
          for (size_t i = 0; i < 8; i++)
          {
            for (size_t j = i + 1; j < 8; j++)
            {
              {
                const float tmp = intermediateBufferF[i * 16 + j];
                intermediateBufferF[i * 16 + j] = intermediateBufferF[j * 16 + i];
                intermediateBufferF[j * 16 + i] = tmp;
              }

              {
                const float tmp = intermediateBufferF[i * 16 + j + 8];
                intermediateBufferF[i * 16 + j + 8] = intermediateBufferF[j * 16 + i + 8];
                intermediateBufferF[j * 16 + i + 8] = tmp;
              }
            }
          }
        }

        // Apply discrete cosine transform.
        for (size_t i = 0; i < 8; i++)
          inplace_dct8_sse41(intermediateBuffer + i * 4);

        // Convert & Store.
        {
          // Prepare:
          const __m128i _0xFF = _mm_set1_epi32(0xFF);
          const __m128 _subtract = _mm_set1_ps(subtract);

          for (size_t i = 0; i < 8; i++)
          {
            // Convert:
            // intermediateBuffer[i] = min(0xFF, max(0, (intermediateBuffer[i] * qTable[i] + subtract)));
            intermediateBuffer[i * 4] = _mm_castsi128_ps(_mm_min_epi32(_0xFF, _mm_max_epi32(_mm_setzero_si128(), _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4], qTable[i * 2]), _subtract)))));
            intermediateBuffer[i * 4 + 1] = _mm_castsi128_ps(_mm_min_epi32(_0xFF, _mm_max_epi32(_mm_setzero_si128(), _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4 + 1], qTable[i * 2 + 1]), _subtract)))));

            intermediateBuffer[i * 4 + 2] = _mm_castsi128_ps(_mm_min_epi32(_0xFF, _mm_max_epi32(_mm_setzero_si128(), _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4 + 2], qTable[i * 2]), _subtract)))));
            intermediateBuffer[i * 4 + 3] = _mm_castsi128_ps(_mm_min_epi32(_0xFF, _mm_max_epi32(_mm_setzero_si128(), _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4 + 3], qTable[i * 2 + 1]), _subtract)))));

            // Store:
            // *pOutPositions[i] = (uint8_t)intermediateBuffer[i];

            // Bit shifting these over because they will be packed into the hi part of a uint16_t.
            intermediateBuffer[i * 4 + 2] = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 2]), 8));
            intermediateBuffer[i * 4 + 3] = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 3]), 8));

            // _mm_extract_epi32 is SSE4.1 (!)
            *pOutPositions[i * 8 + 0] = (uint16_t)(_mm_extract_epi32(_mm_castps_si128(intermediateBuffer[i * 4]), 0) | _mm_extract_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 2]), 0));
            *pOutPositions[i * 8 + 1] = (uint16_t)(_mm_extract_epi32(_mm_castps_si128(intermediateBuffer[i * 4]), 1) | _mm_extract_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 2]), 1));
            *pOutPositions[i * 8 + 2] = (uint16_t)(_mm_extract_epi32(_mm_castps_si128(intermediateBuffer[i * 4]), 2) | _mm_extract_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 2]), 2));
            *pOutPositions[i * 8 + 3] = (uint16_t)(_mm_extract_epi32(_mm_castps_si128(intermediateBuffer[i * 4]), 3) | _mm_extract_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 2]), 3));

            ++pOutPositions[i * 8 + 0];
            ++pOutPositions[i * 8 + 1];
            ++pOutPositions[i * 8 + 2];
            ++pOutPositions[i * 8 + 3];

            *pOutPositions[i * 8 + 4] = (uint16_t)(_mm_extract_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 1]), 0) | _mm_extract_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 3]), 0));
            *pOutPositions[i * 8 + 5] = (uint16_t)(_mm_extract_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 1]), 1) | _mm_extract_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 3]), 1));
            *pOutPositions[i * 8 + 6] = (uint16_t)(_mm_extract_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 1]), 2) | _mm_extract_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 3]), 2));
            *pOutPositions[i * 8 + 7] = (uint16_t)(_mm_extract_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 1]), 3) | _mm_extract_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 3]), 3));

            ++pOutPositions[i * 8 + 4];
            ++pOutPositions[i * 8 + 5];
            ++pOutPositions[i * 8 + 6];
            ++pOutPositions[i * 8 + 7];
          }
        }

        pBlockStart += 16;
      }
    }
  };

  uint16_t *pOutPositions[64];

  // Interleaving stereo buffer write positions.
  {
    const size_t componentOffset = (sizeX * sizeY) / 64;

    for (size_t i = 0; i < 64; i++)
      pOutPositions[i] = reinterpret_cast<uint16_t *>(pTo + componentOffset * i);
  }

  const uint8_t *pLine = pFrom;

  for (size_t y = 0; y < sizeY / 2; y += 8)
  {
    if (y * 2 < startY)
    {
      pLine += 8 * sizeX;

      for (size_t i = 0; i < 64; i++)
        pOutPositions[i] += (sizeX / 8); // Two buffers to a uint16_t *

      continue;
    }
    else if (y * 2 > endY)
    {
      break;
    }

    // Left eye.
    {
      const uint8_t *pBlockStart = pLine;
      internal::encode_line(sizeX, pQuantizeLUT, pBlockStart, pOutPositions);
    }

    // Right eye.
    {
      const uint8_t *pBlockStart = pLine + (sizeX * sizeY / 2);
      internal::encode_line(sizeX, pQuantizeLUT, pBlockStart, pOutPositions);
    }

    pLine += 8 * sizeX;
  }
}

// This function is derived from `simdDCT_EncodeQuantizeReorderStereoBuffer_SSE41_Float` by replacing the few replaceable SSE4.1 intrinsics with non-SIMD implementations.
void simdDCT_EncodeQuantizeReorderStereoBuffer_SSE2_Float(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY)
{
  struct internal
  {
    inline static __m128i _mm_cvtepu8_epi32_no_simd(__m128i val) // This could also be archived by `_mm_shuffle_epi8` which requires SSSE3 though.
    {
      union
      {
        __m128i ret;
        int32_t ret_[4];
      };

      uint8_t *pVal = reinterpret_cast<uint8_t *>(&val);

      ret_[0] = (int32_t)pVal[0];
      ret_[1] = (int32_t)pVal[1];
      ret_[2] = (int32_t)pVal[2];
      ret_[3] = (int32_t)pVal[3];

      return ret;
    }

      static inline void encode_line(const size_t sizeX, IN const float *pQuantizeLUT, IN const uint8_t *pBlockStart, IN_OUT uint16_t *pOutPositions[64])
    {
      constexpr float vr = .95f;
      constexpr float subtract = 127.0f;

      _ALIGN(16) __m128 qTable[64 / 4];

      qTable[0] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[0] * vr)), (255.0f / (pQuantizeLUT[1] * vr)), (255.0f / (pQuantizeLUT[2] * vr)), (255.0f / (pQuantizeLUT[3] * vr)));
      qTable[1] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[4] * vr)), (255.0f / (pQuantizeLUT[5] * vr)), (255.0f / (pQuantizeLUT[6] * vr)), (255.0f / (pQuantizeLUT[7] * vr)));
      qTable[2] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[8] * vr)), (255.0f / (pQuantizeLUT[9] * vr)), (255.0f / (pQuantizeLUT[10] * vr)), (255.0f / (pQuantizeLUT[11] * vr)));
      qTable[3] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[12] * vr)), (255.0f / (pQuantizeLUT[13] * vr)), (255.0f / (pQuantizeLUT[14] * vr)), (255.0f / (pQuantizeLUT[15] * vr)));
      qTable[4] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[16] * vr)), (255.0f / (pQuantizeLUT[17] * vr)), (255.0f / (pQuantizeLUT[18] * vr)), (255.0f / (pQuantizeLUT[19] * vr)));
      qTable[5] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[20] * vr)), (255.0f / (pQuantizeLUT[21] * vr)), (255.0f / (pQuantizeLUT[22] * vr)), (255.0f / (pQuantizeLUT[23] * vr)));
      qTable[6] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[24] * vr)), (255.0f / (pQuantizeLUT[25] * vr)), (255.0f / (pQuantizeLUT[26] * vr)), (255.0f / (pQuantizeLUT[27] * vr)));
      qTable[7] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[28] * vr)), (255.0f / (pQuantizeLUT[29] * vr)), (255.0f / (pQuantizeLUT[30] * vr)), (255.0f / (pQuantizeLUT[31] * vr)));
      qTable[8] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[32] * vr)), (255.0f / (pQuantizeLUT[33] * vr)), (255.0f / (pQuantizeLUT[34] * vr)), (255.0f / (pQuantizeLUT[35] * vr)));
      qTable[9] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[36] * vr)), (255.0f / (pQuantizeLUT[37] * vr)), (255.0f / (pQuantizeLUT[38] * vr)), (255.0f / (pQuantizeLUT[39] * vr)));
      qTable[10] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[40] * vr)), (255.0f / (pQuantizeLUT[41] * vr)), (255.0f / (pQuantizeLUT[42] * vr)), (255.0f / (pQuantizeLUT[43] * vr)));
      qTable[11] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[44] * vr)), (255.0f / (pQuantizeLUT[45] * vr)), (255.0f / (pQuantizeLUT[46] * vr)), (255.0f / (pQuantizeLUT[47] * vr)));
      qTable[12] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[48] * vr)), (255.0f / (pQuantizeLUT[49] * vr)), (255.0f / (pQuantizeLUT[50] * vr)), (255.0f / (pQuantizeLUT[51] * vr)));
      qTable[13] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[52] * vr)), (255.0f / (pQuantizeLUT[53] * vr)), (255.0f / (pQuantizeLUT[54] * vr)), (255.0f / (pQuantizeLUT[55] * vr)));
      qTable[14] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[56] * vr)), (255.0f / (pQuantizeLUT[57] * vr)), (255.0f / (pQuantizeLUT[58] * vr)), (255.0f / (pQuantizeLUT[59] * vr)));
      qTable[15] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[60] * vr)), (255.0f / (pQuantizeLUT[61] * vr)), (255.0f / (pQuantizeLUT[62] * vr)), (255.0f / (pQuantizeLUT[63] * vr)));

      _ALIGN(16) __m128i localBuffer[8];
      static_assert(sizeof(localBuffer) == sizeof(uint8_t) * 128, "Invalid Size.");

      union _ALIGN(16)
      {
        __m128 intermediateBuffer[128 / sizeof(__m128i) * sizeof(float)];
        float intermediateBufferF[128];
      };

      static_assert(sizeof(intermediateBuffer) == sizeof(float) * 128, "Invalid Packing");
      static_assert(sizeof(intermediateBufferF) == sizeof(__m128i) * 32, "Invalid Packing");

      for (size_t x = 0; x < sizeX; x += 16)
      {
        // Acquire block.
        {
          for (size_t i = 0; i < 8; i++)
            localBuffer[i] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pBlockStart + sizeX * i));
        }

        // Convert to intermediateBuffer.
        {
          const __m128 inverse_0xFF_float = _mm_set1_ps(1.f / (float)0xFF);

          for (size_t i = 0; i < 8; i++)
          {
            intermediateBuffer[i * 4 + 0] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32_no_simd(localBuffer[i])));
            intermediateBuffer[i * 4 + 1] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32_no_simd(_mm_srli_si128(localBuffer[i], 4))));
            intermediateBuffer[i * 4 + 2] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32_no_simd(_mm_srli_si128(localBuffer[i], 8))));
            intermediateBuffer[i * 4 + 3] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32_no_simd(_mm_srli_si128(localBuffer[i], 12))));
          }
        }

        // Swap dimensions.
        {
          for (size_t i = 0; i < 8; i++)
          {
            for (size_t j = i + 1; j < 8; j++)
            {
              {
                const float tmp = intermediateBufferF[i * 16 + j];
                intermediateBufferF[i * 16 + j] = intermediateBufferF[j * 16 + i];
                intermediateBufferF[j * 16 + i] = tmp;
              }

              {
                const float tmp = intermediateBufferF[i * 16 + j + 8];
                intermediateBufferF[i * 16 + j + 8] = intermediateBufferF[j * 16 + i + 8];
                intermediateBufferF[j * 16 + i + 8] = tmp;
              }
            }
          }
        }

        // Apply discrete cosine transform.
        for (size_t i = 0; i < 8; i++)
          inplace_dct8_sse2(intermediateBuffer + i * 4);

        // Swap dimensions.
        {
          for (size_t i = 0; i < 8; i++)
          {
            for (size_t j = i + 1; j < 8; j++)
            {
              {
                const float tmp = intermediateBufferF[i * 16 + j];
                intermediateBufferF[i * 16 + j] = intermediateBufferF[j * 16 + i];
                intermediateBufferF[j * 16 + i] = tmp;
              }

              {
                const float tmp = intermediateBufferF[i * 16 + j + 8];
                intermediateBufferF[i * 16 + j + 8] = intermediateBufferF[j * 16 + i + 8];
                intermediateBufferF[j * 16 + i + 8] = tmp;
              }
            }
          }
        }

        // Apply discrete cosine transform.
        for (size_t i = 0; i < 8; i++)
          inplace_dct8_sse2(intermediateBuffer + i * 4);

        // Convert & Store.
        {
          // Prepare:
          const __m128 _0xFFf = _mm_set1_ps((float)0xFF);
          const __m128 _subtract = _mm_set1_ps(subtract);

          for (size_t i = 0; i < 8; i++)
          {
            // Convert:
            // intermediateBuffer[i] = min(0xFF, max(0, (intermediateBuffer[i] * qTable[i] + subtract)));
            intermediateBuffer[i * 4] = _mm_castsi128_ps(_mm_cvtps_epi32(_mm_min_ps(_0xFFf, _mm_max_ps(_mm_setzero_ps(), _mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4], qTable[i * 2]), _subtract)))));
            intermediateBuffer[i * 4 + 1] = _mm_castsi128_ps(_mm_cvtps_epi32(_mm_min_ps(_0xFFf, _mm_max_ps(_mm_setzero_ps(), _mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4 + 1], qTable[i * 2 + 1]), _subtract)))));

            intermediateBuffer[i * 4 + 2] = _mm_castsi128_ps(_mm_cvtps_epi32(_mm_min_ps(_0xFFf, _mm_max_ps(_mm_setzero_ps(), _mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4 + 2], qTable[i * 2]), _subtract)))));
            intermediateBuffer[i * 4 + 3] = _mm_castsi128_ps(_mm_cvtps_epi32(_mm_min_ps(_0xFFf, _mm_max_ps(_mm_setzero_ps(), _mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4 + 3], qTable[i * 2 + 1]), _subtract)))));

            // Store:
            // *pOutPositions[i] = (uint8_t)intermediateBuffer[i];

            // Bit shifting these over because they will be packed into the hi part of a uint16_t.
            intermediateBuffer[i * 4 + 2] = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 2]), 8));
            intermediateBuffer[i * 4 + 3] = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 3]), 8));

            *pOutPositions[i * 8 + 0] = (uint16_t)(_mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4]), 0 * 2) | _mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 2]), 0 * 2));
            *pOutPositions[i * 8 + 1] = (uint16_t)(_mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4]), 1 * 2) | _mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 2]), 1 * 2));
            *pOutPositions[i * 8 + 2] = (uint16_t)(_mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4]), 2 * 2) | _mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 2]), 2 * 2));
            *pOutPositions[i * 8 + 3] = (uint16_t)(_mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4]), 3 * 2) | _mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 2]), 3 * 2));

            ++pOutPositions[i * 8 + 0];
            ++pOutPositions[i * 8 + 1];
            ++pOutPositions[i * 8 + 2];
            ++pOutPositions[i * 8 + 3];

            *pOutPositions[i * 8 + 4] = (uint16_t)(_mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 1]), 0 * 2) | _mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 3]), 0 * 2));
            *pOutPositions[i * 8 + 5] = (uint16_t)(_mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 1]), 1 * 2) | _mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 3]), 1 * 2));
            *pOutPositions[i * 8 + 6] = (uint16_t)(_mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 1]), 2 * 2) | _mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 3]), 2 * 2));
            *pOutPositions[i * 8 + 7] = (uint16_t)(_mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 1]), 3 * 2) | _mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 3]), 3 * 2));

            ++pOutPositions[i * 8 + 4];
            ++pOutPositions[i * 8 + 5];
            ++pOutPositions[i * 8 + 6];
            ++pOutPositions[i * 8 + 7];
          }
        }

        pBlockStart += 16;
      }
    }
  };

  uint16_t *pOutPositions[64];

  // Interleaving stereo buffer write positions.
  {
    const size_t componentOffset = (sizeX * sizeY) / 64;

    for (size_t i = 0; i < 64; i++)
      pOutPositions[i] = reinterpret_cast<uint16_t *>(pTo + componentOffset * i);
  }

  const uint8_t *pLine = pFrom;

  for (size_t y = 0; y < sizeY / 2; y += 8)
  {
    if (y * 2 < startY)
    {
      pLine += 8 * sizeX;

      for (size_t i = 0; i < 64; i++)
        pOutPositions[i] += (sizeX / 8); // Two buffers to a uint16_t *

      continue;
    }
    else if (y * 2 > endY)
    {
      break;
    }

    // Left eye.
    {
      const uint8_t *pBlockStart = pLine;
      internal::encode_line(sizeX, pQuantizeLUT, pBlockStart, pOutPositions);
    }

    // Right eye.
    {
      const uint8_t *pBlockStart = pLine + (sizeX * sizeY / 2);
      internal::encode_line(sizeX, pQuantizeLUT, pBlockStart, pOutPositions);
    }

    pLine += 8 * sizeX;
  }
}

// This function is derived from `simdDCT_EncodeQuantizeReorderStereoBuffer_SSE2_Float` by replacing `_mm_cvtepu8_epi32_no_simd` with an SSSE3 alternative.
void simdDCT_EncodeQuantizeReorderStereoBuffer_SSSE3_Float(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY)
{
  struct internal
  {
#ifndef _MSC_VER
    __attribute__((target("ssse3")))
#endif
    inline static __m128i _mm_cvtepu8_epi32_ssse3(__m128i val)
    {
      const __m128i shuffle_mask = _mm_set_epi8_reverse(0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1);
      return _mm_shuffle_epi8(val, shuffle_mask);
    }

#ifndef _MSC_VER
    __attribute__((target("ssse3")))
#endif
      static inline void encode_line(const size_t sizeX, IN const float *pQuantizeLUT, IN const uint8_t *pBlockStart, IN_OUT uint16_t *pOutPositions[64])
    {
      constexpr float vr = .95f;
      constexpr float subtract = 127.0f;

      _ALIGN(16) __m128 qTable[64 / 4];

      qTable[0] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[0] * vr)), (255.0f / (pQuantizeLUT[1] * vr)), (255.0f / (pQuantizeLUT[2] * vr)), (255.0f / (pQuantizeLUT[3] * vr)));
      qTable[1] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[4] * vr)), (255.0f / (pQuantizeLUT[5] * vr)), (255.0f / (pQuantizeLUT[6] * vr)), (255.0f / (pQuantizeLUT[7] * vr)));
      qTable[2] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[8] * vr)), (255.0f / (pQuantizeLUT[9] * vr)), (255.0f / (pQuantizeLUT[10] * vr)), (255.0f / (pQuantizeLUT[11] * vr)));
      qTable[3] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[12] * vr)), (255.0f / (pQuantizeLUT[13] * vr)), (255.0f / (pQuantizeLUT[14] * vr)), (255.0f / (pQuantizeLUT[15] * vr)));
      qTable[4] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[16] * vr)), (255.0f / (pQuantizeLUT[17] * vr)), (255.0f / (pQuantizeLUT[18] * vr)), (255.0f / (pQuantizeLUT[19] * vr)));
      qTable[5] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[20] * vr)), (255.0f / (pQuantizeLUT[21] * vr)), (255.0f / (pQuantizeLUT[22] * vr)), (255.0f / (pQuantizeLUT[23] * vr)));
      qTable[6] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[24] * vr)), (255.0f / (pQuantizeLUT[25] * vr)), (255.0f / (pQuantizeLUT[26] * vr)), (255.0f / (pQuantizeLUT[27] * vr)));
      qTable[7] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[28] * vr)), (255.0f / (pQuantizeLUT[29] * vr)), (255.0f / (pQuantizeLUT[30] * vr)), (255.0f / (pQuantizeLUT[31] * vr)));
      qTable[8] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[32] * vr)), (255.0f / (pQuantizeLUT[33] * vr)), (255.0f / (pQuantizeLUT[34] * vr)), (255.0f / (pQuantizeLUT[35] * vr)));
      qTable[9] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[36] * vr)), (255.0f / (pQuantizeLUT[37] * vr)), (255.0f / (pQuantizeLUT[38] * vr)), (255.0f / (pQuantizeLUT[39] * vr)));
      qTable[10] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[40] * vr)), (255.0f / (pQuantizeLUT[41] * vr)), (255.0f / (pQuantizeLUT[42] * vr)), (255.0f / (pQuantizeLUT[43] * vr)));
      qTable[11] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[44] * vr)), (255.0f / (pQuantizeLUT[45] * vr)), (255.0f / (pQuantizeLUT[46] * vr)), (255.0f / (pQuantizeLUT[47] * vr)));
      qTable[12] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[48] * vr)), (255.0f / (pQuantizeLUT[49] * vr)), (255.0f / (pQuantizeLUT[50] * vr)), (255.0f / (pQuantizeLUT[51] * vr)));
      qTable[13] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[52] * vr)), (255.0f / (pQuantizeLUT[53] * vr)), (255.0f / (pQuantizeLUT[54] * vr)), (255.0f / (pQuantizeLUT[55] * vr)));
      qTable[14] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[56] * vr)), (255.0f / (pQuantizeLUT[57] * vr)), (255.0f / (pQuantizeLUT[58] * vr)), (255.0f / (pQuantizeLUT[59] * vr)));
      qTable[15] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[60] * vr)), (255.0f / (pQuantizeLUT[61] * vr)), (255.0f / (pQuantizeLUT[62] * vr)), (255.0f / (pQuantizeLUT[63] * vr)));

      _ALIGN(16) __m128i localBuffer[8];
      static_assert(sizeof(localBuffer) == sizeof(uint8_t) * 128, "Invalid Size.");

      union _ALIGN(16)
      {
        __m128 intermediateBuffer[128 / sizeof(__m128i) * sizeof(float)];
        float intermediateBufferF[128];
      };

      static_assert(sizeof(intermediateBuffer) == sizeof(float) * 128, "Invalid Packing");
      static_assert(sizeof(intermediateBufferF) == sizeof(__m128i) * 32, "Invalid Packing");

      for (size_t x = 0; x < sizeX; x += 16)
      {
        // Acquire block.
        {
          for (size_t i = 0; i < 8; i++)
            localBuffer[i] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pBlockStart + sizeX * i));
        }

        // Convert to intermediateBuffer.
        {
          const __m128 inverse_0xFF_float = _mm_set1_ps(1.f / (float)0xFF);

          for (size_t i = 0; i < 8; i++)
          {
            intermediateBuffer[i * 4 + 0] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32_ssse3(localBuffer[i])));
            intermediateBuffer[i * 4 + 1] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32_ssse3(_mm_srli_si128(localBuffer[i], 4))));
            intermediateBuffer[i * 4 + 2] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32_ssse3(_mm_srli_si128(localBuffer[i], 8))));
            intermediateBuffer[i * 4 + 3] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32_ssse3(_mm_srli_si128(localBuffer[i], 12))));
          }
        }

        // Swap dimensions.
        {
          for (size_t i = 0; i < 8; i++)
          {
            for (size_t j = i + 1; j < 8; j++)
            {
              {
                const float tmp = intermediateBufferF[i * 16 + j];
                intermediateBufferF[i * 16 + j] = intermediateBufferF[j * 16 + i];
                intermediateBufferF[j * 16 + i] = tmp;
              }

              {
                const float tmp = intermediateBufferF[i * 16 + j + 8];
                intermediateBufferF[i * 16 + j + 8] = intermediateBufferF[j * 16 + i + 8];
                intermediateBufferF[j * 16 + i + 8] = tmp;
              }
            }
          }
        }

        // Apply discrete cosine transform.
        for (size_t i = 0; i < 8; i++)
          inplace_dct8_sse2(intermediateBuffer + i * 4);

        // Swap dimensions.
        {
          for (size_t i = 0; i < 8; i++)
          {
            for (size_t j = i + 1; j < 8; j++)
            {
              {
                const float tmp = intermediateBufferF[i * 16 + j];
                intermediateBufferF[i * 16 + j] = intermediateBufferF[j * 16 + i];
                intermediateBufferF[j * 16 + i] = tmp;
              }

              {
                const float tmp = intermediateBufferF[i * 16 + j + 8];
                intermediateBufferF[i * 16 + j + 8] = intermediateBufferF[j * 16 + i + 8];
                intermediateBufferF[j * 16 + i + 8] = tmp;
              }
            }
          }
        }

        // Apply discrete cosine transform.
        for (size_t i = 0; i < 8; i++)
          inplace_dct8_sse2(intermediateBuffer + i * 4);

        // Convert & Store.
        {
          // Prepare:
          const __m128 _0xFFf = _mm_set1_ps((float)0xFF);
          const __m128 _subtract = _mm_set1_ps(subtract);

          for (size_t i = 0; i < 8; i++)
          {
            // Convert:
            // intermediateBuffer[i] = min(0xFF, max(0, (intermediateBuffer[i] * qTable[i] + subtract)));
            intermediateBuffer[i * 4] = _mm_castsi128_ps(_mm_cvtps_epi32(_mm_min_ps(_0xFFf, _mm_max_ps(_mm_setzero_ps(), _mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4], qTable[i * 2]), _subtract)))));
            intermediateBuffer[i * 4 + 1] = _mm_castsi128_ps(_mm_cvtps_epi32(_mm_min_ps(_0xFFf, _mm_max_ps(_mm_setzero_ps(), _mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4 + 1], qTable[i * 2 + 1]), _subtract)))));

            intermediateBuffer[i * 4 + 2] = _mm_castsi128_ps(_mm_cvtps_epi32(_mm_min_ps(_0xFFf, _mm_max_ps(_mm_setzero_ps(), _mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4 + 2], qTable[i * 2]), _subtract)))));
            intermediateBuffer[i * 4 + 3] = _mm_castsi128_ps(_mm_cvtps_epi32(_mm_min_ps(_0xFFf, _mm_max_ps(_mm_setzero_ps(), _mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4 + 3], qTable[i * 2 + 1]), _subtract)))));

            // Store:
            // *pOutPositions[i] = (uint8_t)intermediateBuffer[i];

            // Bit shifting these over because they will be packed into the hi part of a uint16_t.
            intermediateBuffer[i * 4 + 2] = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 2]), 8));
            intermediateBuffer[i * 4 + 3] = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(intermediateBuffer[i * 4 + 3]), 8));

            *pOutPositions[i * 8 + 0] = (uint16_t)(_mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4]), 0 * 2) | _mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 2]), 0 * 2));
            *pOutPositions[i * 8 + 1] = (uint16_t)(_mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4]), 1 * 2) | _mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 2]), 1 * 2));
            *pOutPositions[i * 8 + 2] = (uint16_t)(_mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4]), 2 * 2) | _mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 2]), 2 * 2));
            *pOutPositions[i * 8 + 3] = (uint16_t)(_mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4]), 3 * 2) | _mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 2]), 3 * 2));

            ++pOutPositions[i * 8 + 0];
            ++pOutPositions[i * 8 + 1];
            ++pOutPositions[i * 8 + 2];
            ++pOutPositions[i * 8 + 3];

            *pOutPositions[i * 8 + 4] = (uint16_t)(_mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 1]), 0 * 2) | _mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 3]), 0 * 2));
            *pOutPositions[i * 8 + 5] = (uint16_t)(_mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 1]), 1 * 2) | _mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 3]), 1 * 2));
            *pOutPositions[i * 8 + 6] = (uint16_t)(_mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 1]), 2 * 2) | _mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 3]), 2 * 2));
            *pOutPositions[i * 8 + 7] = (uint16_t)(_mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 1]), 3 * 2) | _mm_extract_epi16(_mm_castps_si128(intermediateBuffer[i * 4 + 3]), 3 * 2));

            ++pOutPositions[i * 8 + 4];
            ++pOutPositions[i * 8 + 5];
            ++pOutPositions[i * 8 + 6];
            ++pOutPositions[i * 8 + 7];
          }
        }

        pBlockStart += 16;
      }
    }
  };

  uint16_t *pOutPositions[64];

  // Interleaving stereo buffer write positions.
  {
    const size_t componentOffset = (sizeX * sizeY) / 64;

    for (size_t i = 0; i < 64; i++)
      pOutPositions[i] = reinterpret_cast<uint16_t *>(pTo + componentOffset * i);
  }

  const uint8_t *pLine = pFrom;

  for (size_t y = 0; y < sizeY / 2; y += 8)
  {
    if (y * 2 < startY)
    {
      pLine += 8 * sizeX;

      for (size_t i = 0; i < 64; i++)
        pOutPositions[i] += (sizeX / 8); // Two buffers to a uint16_t *

      continue;
    }
    else if (y * 2 > endY)
    {
      break;
    }

    // Left eye.
    {
      const uint8_t *pBlockStart = pLine;
      internal::encode_line(sizeX, pQuantizeLUT, pBlockStart, pOutPositions);
    }

    // Right eye.
    {
      const uint8_t *pBlockStart = pLine + (sizeX * sizeY / 2);
      internal::encode_line(sizeX, pQuantizeLUT, pBlockStart, pOutPositions);
    }

    pLine += 8 * sizeX;
  }
}

//////////////////////////////////////////////////////////////////////////

void simdDCT_EncodeQuantizeBuffer_SSE41_Float(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY)
{
  struct internal
  {
#ifndef _MSC_VER
    __attribute__((target("sse4.1")))
#endif
    static inline void encode_line(const size_t sizeX, IN const float *pQuantizeLUT, IN const uint8_t *pBlockStart, OUT uint8_t *pTo)
    {
      constexpr float vr = .95f;
      constexpr float subtract = 127.0f;

      _ALIGN(16) __m128 qTable[64 / 4];

      qTable[0] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[0] * vr)), (255.0f / (pQuantizeLUT[1] * vr)), (255.0f / (pQuantizeLUT[2] * vr)), (255.0f / (pQuantizeLUT[3] * vr)));
      qTable[1] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[4] * vr)), (255.0f / (pQuantizeLUT[5] * vr)), (255.0f / (pQuantizeLUT[6] * vr)), (255.0f / (pQuantizeLUT[7] * vr)));
      qTable[2] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[8] * vr)), (255.0f / (pQuantizeLUT[9] * vr)), (255.0f / (pQuantizeLUT[10] * vr)), (255.0f / (pQuantizeLUT[11] * vr)));
      qTable[3] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[12] * vr)), (255.0f / (pQuantizeLUT[13] * vr)), (255.0f / (pQuantizeLUT[14] * vr)), (255.0f / (pQuantizeLUT[15] * vr)));
      qTable[4] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[16] * vr)), (255.0f / (pQuantizeLUT[17] * vr)), (255.0f / (pQuantizeLUT[18] * vr)), (255.0f / (pQuantizeLUT[19] * vr)));
      qTable[5] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[20] * vr)), (255.0f / (pQuantizeLUT[21] * vr)), (255.0f / (pQuantizeLUT[22] * vr)), (255.0f / (pQuantizeLUT[23] * vr)));
      qTable[6] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[24] * vr)), (255.0f / (pQuantizeLUT[25] * vr)), (255.0f / (pQuantizeLUT[26] * vr)), (255.0f / (pQuantizeLUT[27] * vr)));
      qTable[7] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[28] * vr)), (255.0f / (pQuantizeLUT[29] * vr)), (255.0f / (pQuantizeLUT[30] * vr)), (255.0f / (pQuantizeLUT[31] * vr)));
      qTable[8] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[32] * vr)), (255.0f / (pQuantizeLUT[33] * vr)), (255.0f / (pQuantizeLUT[34] * vr)), (255.0f / (pQuantizeLUT[35] * vr)));
      qTable[9] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[36] * vr)), (255.0f / (pQuantizeLUT[37] * vr)), (255.0f / (pQuantizeLUT[38] * vr)), (255.0f / (pQuantizeLUT[39] * vr)));
      qTable[10] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[40] * vr)), (255.0f / (pQuantizeLUT[41] * vr)), (255.0f / (pQuantizeLUT[42] * vr)), (255.0f / (pQuantizeLUT[43] * vr)));
      qTable[11] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[44] * vr)), (255.0f / (pQuantizeLUT[45] * vr)), (255.0f / (pQuantizeLUT[46] * vr)), (255.0f / (pQuantizeLUT[47] * vr)));
      qTable[12] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[48] * vr)), (255.0f / (pQuantizeLUT[49] * vr)), (255.0f / (pQuantizeLUT[50] * vr)), (255.0f / (pQuantizeLUT[51] * vr)));
      qTable[13] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[52] * vr)), (255.0f / (pQuantizeLUT[53] * vr)), (255.0f / (pQuantizeLUT[54] * vr)), (255.0f / (pQuantizeLUT[55] * vr)));
      qTable[14] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[56] * vr)), (255.0f / (pQuantizeLUT[57] * vr)), (255.0f / (pQuantizeLUT[58] * vr)), (255.0f / (pQuantizeLUT[59] * vr)));
      qTable[15] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[60] * vr)), (255.0f / (pQuantizeLUT[61] * vr)), (255.0f / (pQuantizeLUT[62] * vr)), (255.0f / (pQuantizeLUT[63] * vr)));

      _ALIGN(16) __m128i localBuffer[8];
      static_assert(sizeof(localBuffer) == sizeof(uint8_t) * 128, "Invalid Size.");

      union _ALIGN(16)
      {
        __m128 intermediateBuffer[128 / sizeof(__m128i) * sizeof(float)];
        float intermediateBufferF[128];
      };

      static_assert(sizeof(intermediateBuffer) == sizeof(float) * 128, "Invalid Packing");
      static_assert(sizeof(intermediateBufferF) == sizeof(__m128i) * 32, "Invalid Packing");


#define _ -1
      const __m128i shuffleMask = _mm_set_epi8(_, _, _, _, _, _, _, _, _, _, _, _, 12, 8, 4, 0);
#undef _

      for (size_t x = 0; x < sizeX; x += 16)
      {
        // Acquire block.
        {
          for (size_t i = 0; i < 8; i++)
            localBuffer[i] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pBlockStart + sizeX * i));
        }

        // Convert to intermediateBuffer.
        {
          const __m128 inverse_0xFF_float = _mm_set1_ps(1.f / (float)0xFF);

          for (size_t i = 0; i < 8; i++)
          {
            intermediateBuffer[i * 4 + 0] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(localBuffer[i]))); // _mm_cvtepu8_epi32 is SSE4.1 (!)
            intermediateBuffer[i * 4 + 1] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(localBuffer[i], 4))));
            intermediateBuffer[i * 4 + 2] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(localBuffer[i], 8))));
            intermediateBuffer[i * 4 + 3] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(localBuffer[i], 12))));
          }
        }

        // Apply discrete cosine transform.
        for (size_t i = 0; i < 8; i++)
          inplace_dct8_sse41(intermediateBuffer + i * 4);

        // Swap dimensions.
        {
          for (size_t i = 0; i < 8; i++)
          {
            for (size_t j = i + 1; j < 8; j++)
            {
              {
                const float tmp = intermediateBufferF[i * 16 + j];
                intermediateBufferF[i * 16 + j] = intermediateBufferF[j * 16 + i];
                intermediateBufferF[j * 16 + i] = tmp;
              }

              {
                const float tmp = intermediateBufferF[i * 16 + j + 8];
                intermediateBufferF[i * 16 + j + 8] = intermediateBufferF[j * 16 + i + 8];
                intermediateBufferF[j * 16 + i + 8] = tmp;
              }
            }
          }
        }

        // Apply discrete cosine transform.
        for (size_t i = 0; i < 8; i++)
          inplace_dct8_sse41(intermediateBuffer + i * 4);

        // Convert & Store.
        {
          // Prepare:
          const __m128i _0xFF = _mm_set1_epi32(0xFF);
          const __m128 _subtract = _mm_set1_ps(subtract);

          for (size_t i = 0; i < 8; i++)
          {
            // Convert:
            // intermediateBuffer[i] = min(0xFF, max(0, (intermediateBuffer[i] * qTable[i] + subtract)));
            const __m128i clamped04 = _mm_min_epi32(_0xFF, _mm_max_epi32(_mm_setzero_si128(), _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4], qTable[i * 2]), _subtract))));
            const __m128i clamped15 = _mm_min_epi32(_0xFF, _mm_max_epi32(_mm_setzero_si128(), _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4 + 1], qTable[i * 2 + 1]), _subtract))));

            const __m128i clamped26 = _mm_min_epi32(_0xFF, _mm_max_epi32(_mm_setzero_si128(), _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4 + 2], qTable[i * 2]), _subtract))));
            const __m128i clamped37 = _mm_min_epi32(_0xFF, _mm_max_epi32(_mm_setzero_si128(), _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4 + 3], qTable[i * 2 + 1]), _subtract))));

            // Extract: (_mm_extract_epi32 is SSE4.1 (!))
            const uint32_t bytes04 = _mm_extract_epi32(_mm_shuffle_epi8(clamped04, shuffleMask), 0);
            const uint32_t bytes15 = _mm_extract_epi32(_mm_shuffle_epi8(clamped15, shuffleMask), 0);
            const uint32_t bytes26 = _mm_extract_epi32(_mm_shuffle_epi8(clamped26, shuffleMask), 0);
            const uint32_t bytes37 = _mm_extract_epi32(_mm_shuffle_epi8(clamped37, shuffleMask), 0);

            // Store:
            // *pTo[i] = (uint8_t)intermediateBuffer[i];
            reinterpret_cast<uint16_t *>(pTo)[0] = (uint16_t)bytes04;
            reinterpret_cast<uint16_t *>(pTo)[1] = (uint16_t)bytes15;
            reinterpret_cast<uint16_t *>(pTo)[2] = (uint16_t)bytes26;
            reinterpret_cast<uint16_t *>(pTo)[3] = (uint16_t)bytes37;

            reinterpret_cast<uint16_t *>(pTo)[64] = (uint16_t)(bytes04 >> 16);
            reinterpret_cast<uint16_t *>(pTo)[65] = (uint16_t)(bytes15 >> 16);
            reinterpret_cast<uint16_t *>(pTo)[66] = (uint16_t)(bytes26 >> 16);
            reinterpret_cast<uint16_t *>(pTo)[67] = (uint16_t)(bytes37 >> 16);

            pTo += 8;
          }
        }

        pTo += 64;
        pBlockStart += 16;
      }
    }
  };

  const uint8_t *pLine = pFrom;

  for (size_t y = 0; y < sizeY / 2; y += 8)
  {
    if (y * 2 < startY)
    {
      pLine += 8 * sizeX;
      pTo += 8 * sizeX;

      continue;
    }
    else if (y * 2 > endY)
    {
      break;
    }

    const uint8_t *pBlockStart = pLine;
    internal::encode_line(sizeX, pQuantizeLUT, pBlockStart, pTo);

    pLine += 8 * sizeX;
    pTo += 8 * sizeX;
  }
}

// This function is derived from `simdDCT_EncodeQuantizeBuffer_SSE41_Float` by extracting 16 bit values instead of 32 bit values.
void simdDCT_EncodeQuantizeBuffer_SSSE3_Float(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY)
{
  struct internal
  {
#ifndef _MSC_VER
    __attribute__((target("ssse3")))
#endif
      static inline void encode_line(const size_t sizeX, IN const float *pQuantizeLUT, IN const uint8_t *pBlockStart, OUT uint8_t *pTo)
    {
      constexpr float vr = .95f;
      constexpr float subtract = 127.0f;

      _ALIGN(16) __m128 qTable[64 / 4];

      qTable[0] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[0] * vr)), (255.0f / (pQuantizeLUT[1] * vr)), (255.0f / (pQuantizeLUT[2] * vr)), (255.0f / (pQuantizeLUT[3] * vr)));
      qTable[1] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[4] * vr)), (255.0f / (pQuantizeLUT[5] * vr)), (255.0f / (pQuantizeLUT[6] * vr)), (255.0f / (pQuantizeLUT[7] * vr)));
      qTable[2] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[8] * vr)), (255.0f / (pQuantizeLUT[9] * vr)), (255.0f / (pQuantizeLUT[10] * vr)), (255.0f / (pQuantizeLUT[11] * vr)));
      qTable[3] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[12] * vr)), (255.0f / (pQuantizeLUT[13] * vr)), (255.0f / (pQuantizeLUT[14] * vr)), (255.0f / (pQuantizeLUT[15] * vr)));
      qTable[4] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[16] * vr)), (255.0f / (pQuantizeLUT[17] * vr)), (255.0f / (pQuantizeLUT[18] * vr)), (255.0f / (pQuantizeLUT[19] * vr)));
      qTable[5] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[20] * vr)), (255.0f / (pQuantizeLUT[21] * vr)), (255.0f / (pQuantizeLUT[22] * vr)), (255.0f / (pQuantizeLUT[23] * vr)));
      qTable[6] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[24] * vr)), (255.0f / (pQuantizeLUT[25] * vr)), (255.0f / (pQuantizeLUT[26] * vr)), (255.0f / (pQuantizeLUT[27] * vr)));
      qTable[7] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[28] * vr)), (255.0f / (pQuantizeLUT[29] * vr)), (255.0f / (pQuantizeLUT[30] * vr)), (255.0f / (pQuantizeLUT[31] * vr)));
      qTable[8] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[32] * vr)), (255.0f / (pQuantizeLUT[33] * vr)), (255.0f / (pQuantizeLUT[34] * vr)), (255.0f / (pQuantizeLUT[35] * vr)));
      qTable[9] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[36] * vr)), (255.0f / (pQuantizeLUT[37] * vr)), (255.0f / (pQuantizeLUT[38] * vr)), (255.0f / (pQuantizeLUT[39] * vr)));
      qTable[10] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[40] * vr)), (255.0f / (pQuantizeLUT[41] * vr)), (255.0f / (pQuantizeLUT[42] * vr)), (255.0f / (pQuantizeLUT[43] * vr)));
      qTable[11] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[44] * vr)), (255.0f / (pQuantizeLUT[45] * vr)), (255.0f / (pQuantizeLUT[46] * vr)), (255.0f / (pQuantizeLUT[47] * vr)));
      qTable[12] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[48] * vr)), (255.0f / (pQuantizeLUT[49] * vr)), (255.0f / (pQuantizeLUT[50] * vr)), (255.0f / (pQuantizeLUT[51] * vr)));
      qTable[13] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[52] * vr)), (255.0f / (pQuantizeLUT[53] * vr)), (255.0f / (pQuantizeLUT[54] * vr)), (255.0f / (pQuantizeLUT[55] * vr)));
      qTable[14] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[56] * vr)), (255.0f / (pQuantizeLUT[57] * vr)), (255.0f / (pQuantizeLUT[58] * vr)), (255.0f / (pQuantizeLUT[59] * vr)));
      qTable[15] = _mm_set_ps_reverse((255.0f / (pQuantizeLUT[60] * vr)), (255.0f / (pQuantizeLUT[61] * vr)), (255.0f / (pQuantizeLUT[62] * vr)), (255.0f / (pQuantizeLUT[63] * vr)));

      _ALIGN(16) __m128i localBuffer[8];
      static_assert(sizeof(localBuffer) == sizeof(uint8_t) * 128, "Invalid Size.");

      union _ALIGN(16)
      {
        __m128 intermediateBuffer[128 / sizeof(__m128i) * sizeof(float)];
        float intermediateBufferF[128];
      };

      static_assert(sizeof(intermediateBuffer) == sizeof(float) * 128, "Invalid Packing");
      static_assert(sizeof(intermediateBufferF) == sizeof(__m128i) * 32, "Invalid Packing");


#define _ -1
      const __m128i shuffleMask = _mm_set_epi8(_, _, _, _, _, _, _, _, _, _, _, _, 12, 8, 4, 0);
#undef _

      for (size_t x = 0; x < sizeX; x += 16)
      {
        // Acquire block.
        {
          for (size_t i = 0; i < 8; i++)
            localBuffer[i] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pBlockStart + sizeX * i));
        }

        // Convert to intermediateBuffer.
        {
          const __m128 inverse_0xFF_float = _mm_set1_ps(1.f / (float)0xFF);

          for (size_t i = 0; i < 8; i++)
          {
            intermediateBuffer[i * 4 + 0] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(localBuffer[i]))); // _mm_cvtepu8_epi32 is SSE4.1 (!)
            intermediateBuffer[i * 4 + 1] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(localBuffer[i], 4))));
            intermediateBuffer[i * 4 + 2] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(localBuffer[i], 8))));
            intermediateBuffer[i * 4 + 3] = _mm_mul_ps(inverse_0xFF_float, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(localBuffer[i], 12))));
          }
        }

        // Apply discrete cosine transform.
        for (size_t i = 0; i < 8; i++)
          inplace_dct8_sse2(intermediateBuffer + i * 4);

        // Swap dimensions.
        {
          for (size_t i = 0; i < 8; i++)
          {
            for (size_t j = i + 1; j < 8; j++)
            {
              {
                const float tmp = intermediateBufferF[i * 16 + j];
                intermediateBufferF[i * 16 + j] = intermediateBufferF[j * 16 + i];
                intermediateBufferF[j * 16 + i] = tmp;
              }

              {
                const float tmp = intermediateBufferF[i * 16 + j + 8];
                intermediateBufferF[i * 16 + j + 8] = intermediateBufferF[j * 16 + i + 8];
                intermediateBufferF[j * 16 + i + 8] = tmp;
              }
            }
          }
        }

        // Apply discrete cosine transform.
        for (size_t i = 0; i < 8; i++)
          inplace_dct8_sse2(intermediateBuffer + i * 4);

        // Convert & Store.
        {
          // Prepare:
          const __m128i _0xFF = _mm_set1_epi32(0xFF);
          const __m128 _subtract = _mm_set1_ps(subtract);

          for (size_t i = 0; i < 8; i++)
          {
            // Convert:
            // intermediateBuffer[i] = min(0xFF, max(0, (intermediateBuffer[i] * qTable[i] + subtract)));
            const __m128i clamped04 = _mm_min_epi32(_0xFF, _mm_max_epi32(_mm_setzero_si128(), _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4], qTable[i * 2]), _subtract))));
            const __m128i clamped15 = _mm_min_epi32(_0xFF, _mm_max_epi32(_mm_setzero_si128(), _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4 + 1], qTable[i * 2 + 1]), _subtract))));
            const __m128i clamped26 = _mm_min_epi32(_0xFF, _mm_max_epi32(_mm_setzero_si128(), _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4 + 2], qTable[i * 2]), _subtract))));
            const __m128i clamped37 = _mm_min_epi32(_0xFF, _mm_max_epi32(_mm_setzero_si128(), _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(intermediateBuffer[i * 4 + 3], qTable[i * 2 + 1]), _subtract))));

            // Store:
            // *pTo[i] = (uint8_t)intermediateBuffer[i];
            reinterpret_cast<uint16_t *>(pTo)[0] = (uint16_t)_mm_extract_epi16(clamped04, 0);
            reinterpret_cast<uint16_t *>(pTo)[1] = (uint16_t)_mm_extract_epi16(clamped15, 0);
            reinterpret_cast<uint16_t *>(pTo)[2] = (uint16_t)_mm_extract_epi16(clamped26, 0);
            reinterpret_cast<uint16_t *>(pTo)[3] = (uint16_t)_mm_extract_epi16(clamped37, 0);

            reinterpret_cast<uint16_t *>(pTo)[64] = (uint16_t)_mm_extract_epi16(clamped04, 1);
            reinterpret_cast<uint16_t *>(pTo)[65] = (uint16_t)_mm_extract_epi16(clamped15, 1);
            reinterpret_cast<uint16_t *>(pTo)[66] = (uint16_t)_mm_extract_epi16(clamped26, 1);
            reinterpret_cast<uint16_t *>(pTo)[67] = (uint16_t)_mm_extract_epi16(clamped37, 1);

            pTo += 8;
          }
        }

        pTo += 64;
        pBlockStart += 16;
      }
    }
  };

  const uint8_t *pLine = pFrom;

  for (size_t y = 0; y < sizeY / 2; y += 8)
  {
    if (y * 2 < startY)
    {
      pLine += 8 * sizeX;
      pTo += 8 * sizeX;

      continue;
    }
    else if (y * 2 > endY)
    {
      break;
    }

    const uint8_t *pBlockStart = pLine;
    internal::encode_line(sizeX, pQuantizeLUT, pBlockStart, pTo);

    pLine += 8 * sizeX;
    pTo += 8 * sizeX;
  }
}
