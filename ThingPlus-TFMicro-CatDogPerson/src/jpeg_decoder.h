/**
 * JPEG Decoder Header
 * 
 * Interface for decoding JPEG images using the picojpeg library
 */

 #ifndef JPEG_DECODER_H
 #define JPEG_DECODER_H
 
 #include <stdint.h>
 #include <stdbool.h>
 
// Maximum decoded image dimensions for full RGB decode helper.
// We now allow 128x128 frames because the main application already allocates
// a 128*128*3 RGB buffer (g_rgb_buffer in main.cpp). Previously this was 64x64
// and caused "JPEG image too large" errors when capturing 128x128 JPEGs.
// NOTE: We removed the unused streaming downscale path to avoid adding another
// 49 KB static buffer. If memory becomes tight, consider either:
//   (A) Reducing kTensorArenaSize a little
//   (B) Capturing at 96x96 (if supported) and keeping these at 128
//   (C) Re‑introducing a streaming MCU -> resized buffer path (no full frame)
#define MAX_DECODED_WIDTH  128
#define MAX_DECODED_HEIGHT 128
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 /**
  * Initialize JPEG decoder with buffer data
  */
 bool jpeg_decoder_init(const uint8_t* jpeg_data, uint32_t jpeg_data_size);
 
 /**
  * Decode JPEG image into RGB buffer
  */
 bool jpeg_decode_to_rgb(const uint8_t* jpeg_data, uint32_t jpeg_data_size, 
                         uint8_t* output_buffer, uint32_t* width, uint32_t* height);
 
// (Removed) jpeg_decode_to_model_input: a streaming/downscale helper that
// required its own intermediate buffer. It was unused in the current
// application and would have increased static RAM when expanding MAX_DECODED_*.
// Re‑add if a memory‑efficient streaming resize path is later required.
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif // JPEG_DECODER_H