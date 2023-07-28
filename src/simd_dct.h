#ifndef simd_dct_h__
#define simd_dct_h__

#include <stdint.h>
#include <stddef.h>

#ifndef IN
#define IN
#endif

#ifndef OUT
#define OUT
#endif

#ifndef IN_OUT
#define IN_OUT IN OUT
#endif

#define _SUCCEEDED(errorCode) (sdr_Success == (errorCode))
#define _FAILED(errorCode) (!(_SUCCEEDED(errorCode)))

enum simdDctResult
{
  sdr_Success,
  sdr_InvalidParameter,
  sdr_NotSupported,
};

simdDctResult simdDCT_EncodeQuantizeBuffer(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY);
simdDctResult simdDCT_EncodeQuantizeReorderStereoBuffer(IN const uint8_t *pFrom, OUT uint8_t *pTo, IN const float *pQuantizeLUT, const size_t sizeX, const size_t sizeY, const size_t startY, const size_t endY);

#endif // simd_dct_h__
