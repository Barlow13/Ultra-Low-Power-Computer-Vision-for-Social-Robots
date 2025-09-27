/**
 * ThingPlus-TFMicro-CatDogPerson (Main Application)
 * ---------------------------------------------------------------------------
 * Dual‑core RP2040 application performing multi‑label classification
 * (person / dog / cat / none) on frames captured by an Arducam Mega module.
 *
 *  Core0 responsibilities:
 *    - Hardware & camera init
 *    - Frame acquisition (JPEG)
 *    - Shell / logging / scheduling
 *
 *  Core1 responsibilities:
 *    - TensorFlow Lite Micro model init
 *    - JPEG decode + resize + pre‑processing
 *    - Running inference & post‑processing
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "pico/util/queue.h"
#include "pico/time.h"
#include "hardware/spi.h"
#include "hardware/gpio.h"
#include "hardware/sync.h"
#include "hardware/clocks.h"
#include "Arducam/Arducam_Mega.h"
#include "jpeg_decoder.h" 
#include "picojpeg.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#ifndef APP_SPI_BAUD
#define APP_SPI_BAUD        (8 * 1000 * 1000u)   // 8 MHz
#endif
#define SPI_PORT            spi0
#define PIN_SCK             2
#define PIN_COPI            3
#define PIN_CIPO            4
#define PIN_CS              16

#define APP_DEBUG           1
#define APP_CAPTURE_JPEG_MODE CAM_IMAGE_MODE_96X96
#define APP_CAPTURE_PIX_FMT   CAM_IMAGE_PIX_FMT_JPG

static constexpr size_t kTensorArenaSize = 170 * 1024;

// Enable one-shot arena usage measurement (set to 1 temporarily). After you
// read the printed value, set back to 0 and optionally reduce kTensorArenaSize.
#ifndef APP_ARENA_MEASURE
#define APP_ARENA_MEASURE 0
#endif

// Maximum model input supported by static work buffer (RGB)
static constexpr int kMaxInputWidth  = 128;
static constexpr int kMaxInputHeight = 128;

// JPEG capture buffer size
static constexpr size_t kMaxJpegSize = 16 * 1024;

// Inference pacing
static constexpr uint32_t kLoopDelayMs = 500;  // reduced delay between captures (was 2000)
// Enable detailed timing (set to 0 to disable quickly)
#ifndef APP_ENABLE_TIMING
#define APP_ENABLE_TIMING 1
#endif

#if APP_ENABLE_TIMING
struct TimingStats {
    uint32_t capture_ms = 0;
    uint32_t decode_ms = 0;
    uint32_t resize_ms = 0;
    uint32_t invoke_ms = 0;
    uint32_t frames = 0;
    void accumulate(uint32_t c, uint32_t d, uint32_t r, uint32_t i) {
        capture_ms += c; decode_ms += d; resize_ms += r; invoke_ms += i; ++frames; }
    void print_and_reset() {
        if (frames == 0) return;
        printf("[TIMING] avg over %lu frames | capture=%lu ms decode=%lu ms resize=%lu ms invoke=%lu ms total=%lu ms\n",
               (unsigned long)frames,
               (unsigned long)(capture_ms/frames),
               (unsigned long)(decode_ms/frames),
               (unsigned long)(resize_ms/frames),
               (unsigned long)(invoke_ms/frames),
               (unsigned long)((capture_ms+decode_ms+resize_ms+invoke_ms)/frames));
        capture_ms = decode_ms = resize_ms = invoke_ms = frames = 0;
    }
};
static TimingStats g_timing;             // shared summary (core0 collects capture; core1 collects rest)
static volatile uint32_t g_last_decode_ms = 0;
static volatile uint32_t g_last_resize_ms = 0;
#endif
static constexpr uint32_t kInitTimeoutMs = 10000;
static constexpr uint32_t kInvokeTimeoutMs = 5000;

// Class metadata
static constexpr int   kNumClasses = 4;
static constexpr float kClassThresholds[kNumClasses] = {0.55f, 0.30f, 0.35f, 0.25f};
static constexpr const char* kClassNames[kNumClasses] = {"person", "dog", "cat", "none"};

// ---------------------------------------------------------------------------
// Data structures & shared state
// ---------------------------------------------------------------------------

enum CoreCommand : uint32_t {
    CMD_PROCESS_IMAGE      = 1,
    CMD_DONE               = 2,
    CMD_ERROR              = 3,
};

struct InferenceResult {
    float    scores[kNumClasses];
    uint8_t  predictions[kNumClasses];
    uint32_t inference_time_ms;
    bool     valid;
};

// Shared objects
static queue_t    q_core0_to_core1;
static queue_t    q_core1_to_core0;
static spin_lock_t* g_spinlock;
static volatile InferenceResult g_shared_result{};

// Buffers (cache‑line aligned for safety)
alignas(8) static uint8_t g_jpeg_buffer[kMaxJpegSize];
alignas(8) static uint8_t g_rgb_buffer[kMaxInputWidth * kMaxInputHeight * 3];
static volatile uint32_t g_jpeg_size = 0;

// Model state (core1 only after init)
namespace {
    tflite::MicroErrorReporter  s_micro_reporter;
    tflite::ErrorReporter*      s_reporter = &s_micro_reporter;
    const tflite::Model*        s_model = nullptr;
    tflite::MicroInterpreter*   s_interpreter = nullptr;
    TfLiteTensor*               s_input = nullptr;
    TfLiteTensor*               s_output = nullptr;
    alignas(16) static uint8_t  s_tensor_arena[kTensorArenaSize];
}

// Dynamic (post‑init) model input shape
static volatile int g_input_w = 96;
static volatile int g_input_h = 96;
static volatile int g_input_c = 3;

// Camera instance
static Arducam_Mega g_camera(PIN_CS);

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

static inline void dbg(const char* msg) {
#if APP_DEBUG
    printf("%s\n", msg);
#endif
}

static void print_result(const InferenceResult& r) {
    if (!r.valid) { printf("No result available\n"); return; }
    printf("\n--- Inference (%lu ms) ---\n", (unsigned long)r.inference_time_ms);
    bool any = false;
    for (int i = 0; i < kNumClasses; ++i) {
        bool det = r.predictions[i] != 0;
        printf("  %s: %.3f%s\n", kClassNames[i], r.scores[i], det ? " *" : "");
        any |= det;
    }
    if (!any) printf("  (no class exceeded threshold)\n");
    printf("---------------------------\n");
}

static void postprocess_output(const TfLiteTensor* out, InferenceResult* res) {
    if (!out || !res) return;
    res->valid = false;
    switch (out->type) {
        case kTfLiteUInt8: {
            const uint8_t* d = out->data.uint8;
            for (int i = 0; i < kNumClasses; ++i) {
                float p = d[i] / 255.f;
                res->scores[i] = p;
                res->predictions[i] = p > kClassThresholds[i];
            }
            break;
        }
        case kTfLiteInt8: {
            const int8_t* d = out->data.int8;
            float s = out->params.scale; int zp = out->params.zero_point;
            for (int i = 0; i < kNumClasses; ++i) {
                float p = s * (d[i] - zp);
                if (p < 0.f) p = 0.f; else if (p > 1.f) p = 1.f;
                res->scores[i] = p;
                res->predictions[i] = p > kClassThresholds[i];
            }
            break;
        }
        case kTfLiteFloat32: {
            const float* d = out->data.f;
            for (int i = 0; i < kNumClasses; ++i) {
                float p = d[i];
                if (p < 0.f) p = 0.f; else if (p > 1.f) p = 1.f;
                res->scores[i] = p;
                res->predictions[i] = p > kClassThresholds[i];
            }
            break;
        }
        default:
            for (int i = 0; i < kNumClasses; ++i) { res->scores[i] = 0.f; res->predictions[i] = 0; }
            break;
    }
    res->valid = true;
}

static void print_free_ram() {
    extern char __StackLimit, __bss_end__;
    printf("Free RAM (approx): %d bytes\n", (int)(&__StackLimit - &__bss_end__));
}

// ---------------------------------------------------------------------------
// Hardware & camera setup (Core0)
// ---------------------------------------------------------------------------

static bool init_hardware() {
    stdio_init_all();
    sleep_ms(600); // allow USB CDC to enumerate
    set_sys_clock_khz(133000, true);
    dbg("[HW] Clock set to 133 MHz");

    spi_init(SPI_PORT, APP_SPI_BAUD);
    gpio_set_function(PIN_SCK,  GPIO_FUNC_SPI);
    gpio_set_function(PIN_COPI, GPIO_FUNC_SPI);
    gpio_set_function(PIN_CIPO, GPIO_FUNC_SPI);
    gpio_init(PIN_CS);
    gpio_set_dir(PIN_CS, GPIO_OUT);
    gpio_put(PIN_CS, 1);

    g_spinlock = spin_lock_init(0);
    queue_init(&q_core0_to_core1, sizeof(uint32_t), 4);
    queue_init(&q_core1_to_core0, sizeof(uint32_t), 4);
    dbg("[HW] Queues + spinlock ready");
    return true;
}

static bool init_camera() {
    dbg("[CAM] Initializing...");
    g_camera.begin();
    g_camera.setRotation(CAM_ROTATION_180);
    g_camera.setAutoFocus(1);
    g_camera.setAutoExposure(1);
    g_camera.setAutoWhiteBalance(1);
    g_camera.setAutoISOSensitive(1);
    dbg("[CAM] Ready");
    return true;
}

// Capture JPEG into shared buffer (Core0)
static bool capture_jpeg(uint32_t* elapsed_ms_out) {
#if APP_ENABLE_TIMING
    absolute_time_t t0 = get_absolute_time();
#endif
    CamStatus st = g_camera.takePicture(APP_CAPTURE_JPEG_MODE, APP_CAPTURE_PIX_FMT);
    if (st != CAM_ERR_SUCCESS) { printf("[CAM] takePicture failed (%d)\n", st); return false; }
    uint32_t len = g_camera.getTotalLength();
    if (len == 0 || len > kMaxJpegSize) { printf("[CAM] Bad JPEG size %lu\n", len); return false; }
    uint32_t read = 0;
    while (read < len) {
        uint8_t chunk = 128; if (len - read < chunk) chunk = len - read; if (chunk > 255) chunk = 255; // safety
        read += g_camera.readBuff(g_jpeg_buffer + read, chunk);
    }
    g_jpeg_size = len;
    printf("[CAM] Captured %lu bytes\n", (unsigned long)len);
#if APP_ENABLE_TIMING
    if (elapsed_ms_out) *elapsed_ms_out = absolute_time_diff_us(t0, get_absolute_time())/1000;
#endif
    return true;
}

// ---------------------------------------------------------------------------
// Image processing + input tensor population (Core1)
// ---------------------------------------------------------------------------

static bool decode_and_resize_to_model(uint8_t* rgb_out, int target_w, int target_h) {
    uint32_t dec_w = 0, dec_h = 0;
#if APP_ENABLE_TIMING
    absolute_time_t t_dec0 = get_absolute_time();
#endif
    if (!jpeg_decode_to_rgb(g_jpeg_buffer, g_jpeg_size, rgb_out, &dec_w, &dec_h)) {
        printf("[ML] JPEG decode failed\n"); return false; }
#if APP_ENABLE_TIMING
    g_last_decode_ms = absolute_time_diff_us(t_dec0, get_absolute_time())/1000;
#endif
    if ((int)dec_w == target_w && (int)dec_h == target_h) return true;
    if (target_w > kMaxInputWidth || target_h > kMaxInputHeight) {
        printf("[ML] Target dims exceed buffer (%dx%d)\n", target_w, target_h); return false; }
    // In-place nearest-neighbour downscale WITHOUT extra large buffers.
    // Safe because for dec_w>target_w and dec_h>target_h each source index >= dest index
    // (due to sx>=x and sy>=y with integer floor after scaling). We iterate forward.
    const int src_w = (int)dec_w;
    const int src_h = (int)dec_h;
#if APP_ENABLE_TIMING
    absolute_time_t t_res0 = get_absolute_time();
#endif
    for (int y = 0; y < target_h; ++y) {
        int sy = y * src_h / target_h; if (sy >= src_h) sy = src_h - 1;
        for (int x = 0; x < target_w; ++x) {
            int sx = x * src_w / target_w; if (sx >= src_w) sx = src_w - 1;
            int sidx = (sy * src_w + sx) * 3;
            int didx = (y * target_w + x) * 3;
            rgb_out[didx+0] = rgb_out[sidx+0];
            rgb_out[didx+1] = rgb_out[sidx+1];
            rgb_out[didx+2] = rgb_out[sidx+2];
        }
    }
#if APP_ENABLE_TIMING
    g_last_resize_ms = absolute_time_diff_us(t_res0, get_absolute_time())/1000;
#endif
    return true;
}

static bool populate_input(const uint8_t* rgb) {
    if (!s_input) return false;
    const int h = s_input->dims->data[1];
    const int w = s_input->dims->data[2];
    const int c = s_input->dims->data[3];
    if (c != 3) { printf("[ML] Unsupported channel count %d\n", c); return false; }
    g_input_h = h; g_input_w = w; g_input_c = c;
    switch (s_input->type) {
        case kTfLiteFloat32: {
            float* dst = s_input->data.f;
            for (int i = 0; i < h * w * c; ++i) dst[i] = rgb[i] / 255.f;
            break;
        }
        case kTfLiteInt8: {
            int8_t* dst = s_input->data.int8; float sc = s_input->params.scale; int zp = s_input->params.zero_point;
            if (sc == 0) { printf("[ML] Invalid int8 scale=0\n"); return false; }
            for (int i = 0; i < h * w * c; ++i) {
                float v = rgb[i] / 255.f; int32_t q = (int32_t)(v / sc) + zp; if (q < -128) q = -128; if (q > 127) q = 127; dst[i] = (int8_t)q; }
            break;
        }
        case kTfLiteUInt8: {
            uint8_t* dst = s_input->data.uint8; float sc = s_input->params.scale; int zp = s_input->params.zero_point;
            if (sc == 0) { memcpy(dst, rgb, h * w * c); }
            else {
                for (int i = 0; i < h * w * c; ++i) {
                    float v = rgb[i] / 255.f; int32_t q = (int32_t)(v / sc) + zp; if (q < 0) q = 0; if (q > 255) q = 255; dst[i] = (uint8_t)q; }
            }
            break;
        }
        default: printf("[ML] Unsupported input type %d\n", s_input->type); return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Core1 entry (ML pipeline)
// ---------------------------------------------------------------------------
static void core1_entry() {
    printf("[ML] Core1 starting...\n");
    if (model_data_len == 0 || model_data == nullptr) { printf("[ML] No model data\n"); uint32_t e = CMD_ERROR; queue_try_add(&q_core1_to_core0, &e); return; }
    s_model = tflite::GetModel(model_data);
    if (!s_model) { printf("[ML] GetModel failed\n"); uint32_t e = CMD_ERROR; queue_try_add(&q_core1_to_core0, &e); return; }

    static tflite::MicroMutableOpResolver<16> resolver; // add only what we need
    resolver.AddConv2D(); resolver.AddDepthwiseConv2D(); resolver.AddFullyConnected();
    resolver.AddReshape(); resolver.AddSoftmax(); resolver.AddAdd(); resolver.AddMul();
    resolver.AddAveragePool2D(); resolver.AddMaxPool2D(); resolver.AddMean();
    resolver.AddQuantize(); resolver.AddDequantize(); resolver.AddPad(); resolver.AddConcatenation();
    resolver.AddRelu6(); resolver.AddLogistic();

    // IMPORTANT: For arena usage measurement we must pattern fill the arena
    // BEFORE constructing the MicroInterpreter. Its constructor places internal
    // allocator/graph planner objects inside the arena. Previously we filled
    // after construction which overwrote those objects and caused AllocateTensors()
    // to hang, leading to the core0 init timeout you observed.
#if APP_ARENA_MEASURE
    memset((void*)s_tensor_arena, 0xCD, kTensorArenaSize); // fill with canary pattern
#endif
    static tflite::MicroInterpreter static_interpreter(s_model, resolver, s_tensor_arena, kTensorArenaSize, s_reporter);
    s_interpreter = &static_interpreter;
    if (s_interpreter->AllocateTensors() != kTfLiteOk) { printf("[ML] AllocateTensors failed\n"); uint32_t e = CMD_ERROR; queue_try_add(&q_core1_to_core0, &e); return; }
#if APP_ARENA_MEASURE
    // Scan from end backward for first non-pattern byte.
    int used = 0;
    for (int i = (int)kTensorArenaSize - 1; i >= 0; --i) {
        if (s_tensor_arena[i] != (uint8_t)0xCD) { used = i + 1; break; }
    }
    printf("[ML] Arena used ≈ %d bytes (of %u)\n", used, (unsigned)kTensorArenaSize);
    printf("[ML] Suggest new arena: %d KB (used + 8KB margin)\n", (used + 8192 + 1023)/1024);
#endif
    s_input  = s_interpreter->input(0);
    s_output = s_interpreter->output(0);
    if (!s_input || !s_output) { printf("[ML] Missing tensors\n"); uint32_t e = CMD_ERROR; queue_try_add(&q_core1_to_core0, &e); return; }

    if (s_input->dims->size == 4) { g_input_h = s_input->dims->data[1]; g_input_w = s_input->dims->data[2]; g_input_c = s_input->dims->data[3]; }
    printf("[ML] Model loaded (%u bytes) input=%dx%dx%d type=%d output_type=%d\n", model_data_len, g_input_w, g_input_h, g_input_c, s_input->type, s_output->type);
    uint32_t ok = CMD_DONE; queue_try_add(&q_core1_to_core0, &ok);

    // Command loop
    uint32_t cmd;
    while (true) {
        if (queue_try_remove(&q_core0_to_core1, &cmd)) {
            if (cmd == CMD_PROCESS_IMAGE) {
                // Critical section: copy size and decode
                uint32_t save = spin_lock_blocking(g_spinlock);
                uint32_t local_size = g_jpeg_size; (void)local_size; // size already validated on capture
                spin_unlock(g_spinlock, save);

                if (!decode_and_resize_to_model(g_rgb_buffer, g_input_w, g_input_h)) { uint32_t e = CMD_ERROR; queue_try_add(&q_core1_to_core0, &e); continue; }
                if (!populate_input(g_rgb_buffer)) { uint32_t e = CMD_ERROR; queue_try_add(&q_core1_to_core0, &e); continue; }

                absolute_time_t t0 = get_absolute_time();
                TfLiteStatus st = s_interpreter->Invoke();
                uint32_t elapsed = absolute_time_diff_us(t0, get_absolute_time()) / 1000;
                if (st != kTfLiteOk) { printf("[ML] Invoke failed (%d)\n", st); uint32_t e = CMD_ERROR; queue_try_add(&q_core1_to_core0, &e); continue; }

                save = spin_lock_blocking(g_spinlock);
                InferenceResult* r = (InferenceResult*)&g_shared_result;
                r->inference_time_ms = elapsed;
#if APP_ENABLE_TIMING
                // accumulate decode/resize + invoke inside core1 to avoid sync overhead
                g_timing.decode_ms += g_last_decode_ms;
                g_timing.resize_ms += g_last_resize_ms;
                g_timing.invoke_ms += elapsed;
                // capture_ms added on core0 after capture
#endif
                postprocess_output(s_output, r);
                spin_unlock(g_spinlock, save);

                uint32_t d = CMD_DONE; queue_try_add(&q_core1_to_core0, &d);
            }
        }
        sleep_us(80); // yield
    }
}

// ---------------------------------------------------------------------------
// Application (Core0)
// ---------------------------------------------------------------------------
int main() {
    g_shared_result.valid = false;
    if (!init_hardware()) { printf("[APP] Hardware init failed\n"); while (true) sleep_ms(1000); }
    if (!init_camera())   { printf("[APP] Camera init failed\n"); while (true) sleep_ms(1000); }
    print_free_ram();

    multicore_launch_core1(core1_entry);
    printf("[APP] Waiting for model init...\n");
    absolute_time_t init_to = make_timeout_time_ms(kInitTimeoutMs);
    bool ready = false; uint32_t resp;
    while (!ready) {
        if (queue_try_remove(&q_core1_to_core0, &resp)) {
            if (resp == CMD_DONE) ready = true; else if (resp == CMD_ERROR) { printf("[APP] Model init error\n"); while (true) sleep_ms(1000);} }
        if (absolute_time_diff_us(get_absolute_time(), init_to) <= 0) { printf("[APP] Init timeout\n"); while (true) sleep_ms(1000);} 
        sleep_ms(10);
    }
    printf("[APP] Model ready. Entering loop...\n");

    while (true) {
    uint32_t cap_ms = 0;
    if (!capture_jpeg(&cap_ms)) { sleep_ms(500); continue; }
#if APP_ENABLE_TIMING
    g_timing.capture_ms += cap_ms;
#endif
        uint32_t save = spin_lock_blocking(g_spinlock);
        spin_unlock(g_spinlock, save);
        uint32_t c = CMD_PROCESS_IMAGE; queue_try_add(&q_core0_to_core1, &c);

        bool done = false; bool err = false; absolute_time_t to = make_timeout_time_ms(kInvokeTimeoutMs);
        while (!done && !err) {
            if (queue_try_remove(&q_core1_to_core0, &resp)) { if (resp == CMD_DONE) done = true; else if (resp == CMD_ERROR) err = true; }
            if (absolute_time_diff_us(get_absolute_time(), to) <= 0) { printf("[APP] Inference timeout\n"); err = true; }
            sleep_ms(8);
        }
        if (!err) {
            save = spin_lock_blocking(g_spinlock);
            InferenceResult local; 
            local.inference_time_ms = g_shared_result.inference_time_ms;
            local.valid = g_shared_result.valid;
            for (int i = 0; i < kNumClasses; ++i) { local.scores[i] = g_shared_result.scores[i]; local.predictions[i] = g_shared_result.predictions[i]; }
            spin_unlock(g_spinlock, save);
            print_result(local);
#if APP_ENABLE_TIMING
            g_timing.frames++;
            if (g_timing.frames % 8 == 0) { g_timing.print_and_reset(); }
#endif
        } else {
            printf("[APP] Inference error, retrying...\n");
        }
        sleep_ms(kLoopDelayMs);
    }
    return 0;
}