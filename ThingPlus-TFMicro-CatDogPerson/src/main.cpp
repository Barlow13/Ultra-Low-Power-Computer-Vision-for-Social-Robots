/**
 * Brady Barlow
 * Oklahoma State University
 * 11/28/2025
 * ---------------------------------------------------------------------------
 * ThingPlus-TFMicro-CatDogPerson (Main Application)
 * ---------------------------------------------------------------------------
 * Dual-core RP2040 application performing multi-label classification,
 * allowing simultaneous detection of multiple classes (person / dog / cat / none)
 * on frames captured by an Arducam Mega module.
 *
 *  Core0 responsibilities:
 *    - Hardware & camera init
 *    - Frame acquisition (JPEG)
 *    - Shell / logging / scheduling
 *    - GPIO monitoring (PIN17) with IRQ-based edge detection & debouncing
 *    - Optional gating: pauses capture when PIN17 is LOW
 *    - OLED display control (optional, via APP_OLED_ENABLE)
 *
 *  Core1 responsibilities:
 *    - TensorFlow Lite Micro model init
 *    - JPEG decode + resize + pre-processing
 *    - Running inference & post-processing
 *
 *  Features:
 *    - PIN17 gating logic (APP_GATE_BY_PIN17)
 *    - PIN17 debouncing (APP_PIN17_DEBOUNCE_MS)
 *    - SSD1306 OLED support with runtime power control ('o' key toggle)
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
#include "hardware/vreg.h"
#include "Arducam/Arducam_Mega.h"
#include "jpeg_decoder.h" 
#include "picojpeg.h"
#include "model_data.h"
#ifdef APP_OLED_ENABLE
#include "oled_ssd1306.h"
#endif
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#ifndef APP_SPI_BAUD
#define APP_SPI_BAUD        (16 * 1000 * 1000u)   // 16 MHz
#endif
#define SPI_PORT            spi0
#define PIN_SCK             2
#define PIN_COPI            3
#define PIN_CIPO            4
#define PIN_CS              16
// Optional monitor pin (user request): observe transitions on GPIO 17
#define PIN_MONITOR         17

// Enable/disable gating the capture/inference loop based on PIN17 (monitor pin) level.
// 0 = always run (current behavior). 1 = only capture when PIN17 is HIGH. If it goes LOW mid-processing,
//     the current inference finishes, then the system pauses until PIN17 returns HIGH.
#ifndef APP_GATE_BY_PIN17
#define APP_GATE_BY_PIN17 1
#endif

// Debounce interval (milliseconds) for PIN17 edge logging & gating logic. Set to 0 to disable.
#ifndef APP_PIN17_DEBOUNCE_MS
#define APP_PIN17_DEBOUNCE_MS 5
#endif

// Simple ring buffer to transfer IRQ edge events to main loop (avoid printing in ISR)
struct PinEvent { uint32_t ts_ms; uint8_t level; };
static volatile PinEvent g_pin_events[16];
static volatile uint8_t g_pin_evt_head = 0; // next write
static volatile uint8_t g_pin_evt_tail = 0; // next read
static volatile uint8_t g_pin17_level = 0;   // most recent sampled level (updated by ISR)
static volatile uint32_t g_pin17_last_irq_ms = 0; // last accepted IRQ event time
static volatile uint8_t g_pin17_last_irq_level = 255; // last pushed level (255=unset)

static inline bool pin_event_buffer_not_empty() {
    return g_pin_evt_head != g_pin_evt_tail;
}

static inline bool pin_event_buffer_push(uint8_t level) {
    uint8_t next = (uint8_t)((g_pin_evt_head + 1) & 0x0F);
    if (next == g_pin_evt_tail) { return false; } // overflow -> drop
    g_pin_events[g_pin_evt_head].ts_ms = to_ms_since_boot(get_absolute_time());
    g_pin_events[g_pin_evt_head].level = level;
    g_pin_evt_head = next;
    return true;
}

static inline bool pin_event_buffer_pop(PinEvent* out) {
    if (g_pin_evt_head == g_pin_evt_tail) return false;
    // Copy fields individually to avoid volatile assignment ambiguity
    volatile PinEvent* src = &g_pin_events[g_pin_evt_tail];
    out->ts_ms = src->ts_ms;
    out->level = src->level;
    g_pin_evt_tail = (uint8_t)((g_pin_evt_tail + 1) & 0x0F);
    return true;
}

// ISR callback (shared form). We keep it extremely short.
static void pin_irq_callback(uint gpio, uint32_t events) {
    if (gpio == PIN_MONITOR) {
        uint8_t lvl = (uint8_t)gpio_get(PIN_MONITOR);
        g_pin17_level = lvl;
        uint32_t now_ms = to_ms_since_boot(get_absolute_time());
        // Debounce: require time gap AND a change in level
        if (g_pin17_last_irq_level != lvl) {
            if ((APP_PIN17_DEBOUNCE_MS == 0) || (now_ms - g_pin17_last_irq_ms >= (uint32_t)APP_PIN17_DEBOUNCE_MS)) {
                g_pin17_last_irq_level = lvl;
                g_pin17_last_irq_ms = now_ms;
                pin_event_buffer_push(lvl);
            }
        }
    }
}

#define APP_DEBUG           1
#define APP_CAPTURE_JPEG_MODE CAM_IMAGE_MODE_96X96
#define APP_CAPTURE_PIX_FMT   CAM_IMAGE_PIX_FMT_JPG

static constexpr size_t kTensorArenaSize = 155 * 1024;

// Maximum model input supported by static work buffer (RGB)
static constexpr int kMaxInputWidth  = 96;
static constexpr int kMaxInputHeight = 96;

// Maximum camera capture size (must be >= model input)
static constexpr int kMaxCaptureWidth  = 96;
static constexpr int kMaxCaptureHeight = 96;

// JPEG capture buffer size
static constexpr size_t kMaxJpegSize = 8 * 1024;

// Timeouts (milliseconds)
static constexpr uint32_t kInitTimeoutMs = 10000;
static constexpr uint32_t kInvokeTimeoutMs = 5000;

// Class metadata
static constexpr int   kNumClasses = NUM_CLASSES;
static constexpr const float* kClassThresholds = CLASS_THRESHOLDS;
static constexpr const char* const* kClassNames = CLASS_NAMES;
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
alignas(8) static uint8_t g_rgb_buffer[kMaxCaptureWidth * kMaxCaptureHeight * 3];
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
#ifdef APP_OLED_ENABLE
static SSD1306 g_oled;
#endif

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
    
    // Overclock to 240MHz with higher voltage
    vreg_set_voltage(VREG_VOLTAGE_1_25);
    sleep_ms(10);
    set_sys_clock_khz(240000, true);
    dbg("[HW] Clock set to 240 MHz");

    spi_init(SPI_PORT, APP_SPI_BAUD);
    gpio_set_function(PIN_SCK,  GPIO_FUNC_SPI);
    gpio_set_function(PIN_COPI, GPIO_FUNC_SPI);
    gpio_set_function(PIN_CIPO, GPIO_FUNC_SPI);
    gpio_init(PIN_CS);
    gpio_set_dir(PIN_CS, GPIO_OUT);
    gpio_put(PIN_CS, 1);

    // Setup monitor pin as input (no pull). Adjust pulls if your circuit needs it.
    gpio_init(PIN_MONITOR);
    gpio_set_dir(PIN_MONITOR, GPIO_IN);
    // If the line can float, uncomment one of the below:
    // gpio_pull_up(PIN_MONITOR);
    gpio_pull_down(PIN_MONITOR);

    // Install IRQ for both edges (one global callback is fine)
    gpio_set_irq_enabled_with_callback(PIN_MONITOR, GPIO_IRQ_EDGE_RISE | GPIO_IRQ_EDGE_FALL, true, &pin_irq_callback);

    // Initialize cached level
    g_pin17_level = (uint8_t)gpio_get(PIN_MONITOR);


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
    g_camera.setAutoExposure(0);
    g_camera.setAutoWhiteBalance(0);
    g_camera.setAutoISOSensitive(0);
    dbg("[CAM] Ready");
    return true;
}

// Capture JPEG into shared buffer (Core0)
static bool capture_jpeg() {
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
    return true;
}

// ---------------------------------------------------------------------------
// Image processing + input tensor population (Core1)
// ---------------------------------------------------------------------------

static bool decode_and_resize_to_model(uint8_t* rgb_out, int target_w, int target_h) {
    uint32_t dec_w = 0, dec_h = 0;
    if (!jpeg_decode_to_rgb(g_jpeg_buffer, g_jpeg_size, rgb_out, &dec_w, &dec_h)) {
        printf("[ML] JPEG decode failed\n"); return false; }
    if ((int)dec_w == target_w && (int)dec_h == target_h) return true;
    if (target_w > kMaxInputWidth || target_h > kMaxInputHeight) {
        printf("[ML] Target dims exceed buffer (%dx%d)\n", target_w, target_h); return false; }
    // In-place nearest-neighbour downscale WITHOUT extra large buffers.
    // Safe because for dec_w>target_w and dec_h>target_h each source index >= dest index
    // (due to sx>=x and sy>=y with integer floor after scaling). We iterate forward.
    const int src_w = (int)dec_w;
    const int src_h = (int)dec_h;
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
            
            // Optimization: Use a lookup table to avoid 27k+ floating point operations per frame
            // RP2040 has no FPU, so float math is slow.
            static int8_t lookup[256];
            static float last_sc = -1.0f;
            static int last_zp = -9999;
            
            // Rebuild LUT only if params change (or first run)
            if (sc != last_sc || zp != last_zp) {
                for (int i = 0; i < 256; ++i) {
                    float v = i / 255.f;
                    int32_t q = (int32_t)(v / sc) + zp;
                    if (q < -128) q = -128;
                    if (q > 127) q = 127;
                    lookup[i] = (int8_t)q;
                }
                last_sc = sc;
                last_zp = zp;
            }

            for (int i = 0; i < h * w * c; ++i) {
                dst[i] = lookup[rgb[i]];
            }
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

    static tflite::MicroInterpreter static_interpreter(s_model, resolver, s_tensor_arena, kTensorArenaSize, s_reporter);
    s_interpreter = &static_interpreter;
    if (s_interpreter->AllocateTensors() != kTfLiteOk) { printf("[ML] AllocateTensors failed\n"); uint32_t e = CMD_ERROR; queue_try_add(&q_core1_to_core0, &e); return; }
    
    // Print actual arena usage to help optimize kTensorArenaSize
    printf("[ML] Arena used: %u bytes\n", s_interpreter->arena_used_bytes());

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

#ifdef APP_OLED_ENABLE
    // Try init OLED on i2c1: SDA=6, SCL=7 (adjust wiring accordingly)
    if (g_oled.init(i2c1, 6, 7)) {
        g_oled.clear();
        g_oled.drawText(0,0,"TFMicro Ready");
        g_oled.drawText(0,8,"Init...");
        g_oled.flush();
    } else {
        printf("[OLED] init failed\n");
    }
#endif

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
    // Handle simple console commands (non-blocking). Press 'o' to toggle OLED power.
#ifdef APP_OLED_ENABLE
    int ch = getchar_timeout_us(0);
    if (ch == 'o' || ch == 'O') {
        if (g_oled.isPowered()) {
            g_oled.power(false);
            printf("[OLED] OFF\n");
        } else {
            g_oled.power(true);
            printf("[OLED] ON\n");
        }
    }
#endif
    // Optional gating: wait here while PIN17 is LOW (no captures). If high continuously, run continuously.
#if APP_GATE_BY_PIN17
    if (g_pin17_level == 0) {
        static bool announced = false;
        if (!announced) { printf("[GATE] Waiting for PIN17 HIGH to start capture...\n"); announced = true; }
        // Drain and print any pin edge events while waiting
        PinEvent gev; while (pin_event_buffer_pop(&gev)) {
            printf("[PIN17] %s\n", gev.level ? "HIGH" : "LOW");
            if (gev.level) { announced = false; } // will exit gating loop after while condition re-check
        }
        if (g_pin17_level == 0) { sleep_ms(20); continue; }
    }
#endif
    if (!capture_jpeg()) { sleep_ms(500); continue; }
        uint32_t save = spin_lock_blocking(g_spinlock);
        spin_unlock(g_spinlock, save);
        uint32_t c = CMD_PROCESS_IMAGE; queue_try_add(&q_core0_to_core1, &c);

        bool done = false; bool err = false; absolute_time_t to = make_timeout_time_ms(kInvokeTimeoutMs);
        while (!done && !err) {
            if (queue_try_remove(&q_core1_to_core0, &resp)) { if (resp == CMD_DONE) done = true; else if (resp == CMD_ERROR) err = true; }
            if (absolute_time_diff_us(get_absolute_time(), to) <= 0) { printf("[APP] Inference timeout\n"); err = true; }
            // Drain any pin events captured by IRQ
            PinEvent ev; bool printed_initial = false;
            while (pin_event_buffer_pop(&ev)) {
                printf("[PIN17] %s\n", ev.level ? "HIGH" : "LOW");
                printed_initial = true; (void)printed_initial; // (kept if we later add conditional logic)
            }
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
#ifdef APP_OLED_ENABLE
            if (local.valid && g_oled.isPowered()) {
                g_oled.clear();
                
                // Left Column: List classes (Rows 0-2, skip 'none')
                for (int i = 0; i < kNumClasses - 1; ++i) {
                    char buf[20];
                    snprintf(buf, sizeof(buf), "%s:%d%%", kClassNames[i], (int)(local.scores[i] * 100));
                    g_oled.drawText(0, i * 8, buf);
                }

                // Right Column: Info (x=76)
                // Row 0: Time
                char tbuf[16];
                snprintf(tbuf, sizeof(tbuf), "%lums", (unsigned long)local.inference_time_ms);
                g_oled.drawText(76, 0, tbuf);

                // Row 1: Status
                const char* status = "FREE";
#if APP_GATE_BY_PIN17
                status = g_pin17_level ? "RUN" : "WAIT";
#endif
                g_oled.drawText(76, 8, status);

                // Row 3: Branding (Bottom Right)
                g_oled.drawText(44, 24, "BSB-PicoVision");

                g_oled.flush();
            }
#endif
        } else {
            printf("[APP] Inference error, retrying...\n");
        }
        // Post-frame gating: if pin went LOW during processing, pause immediately
#if APP_GATE_BY_PIN17
        if (g_pin17_level == 0) {
            continue; // loop top will block until HIGH again
        }
#endif
    }
    return 0;
}