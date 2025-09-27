/**
 * ThingPlus-TFMicro-CatDogPerson
 *
 * This application uses both RP2040 cores to efficiently run TensorFlow Lite
 * machine learning models for object detection:
 * - Core 0: I/O operations (camera, user interface)
 * - Core 1: TensorFlow Lite inference and image processing
 */

#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "pico/util/queue.h"
#include "hardware/spi.h"
#include "hardware/gpio.h"
#include "hardware/sync.h"
#include "pico/time.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "picojpeg.h"
#include "jpeg_decoder.h"
#include "Arducam/Arducam_Mega.h"

// Project headers - Include the model data
#include "model_data.h"
#include "model_metadata.h"

// TensorFlow Lite includes
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <hardware/clocks.h>
#include <hardware/i2c.h>

#define SPI_PORT spi0
#define PIN_SCK 2
#define PIN_COPI 3
#define PIN_CIPO 4
#define PIN_CS 16
#define DEBUG 1

// Model dimensions are defined in CMakeLists.txt and passed via compiler flags

Arducam_Mega myCAM(PIN_CS);

// Spinlock for shared memory access
spin_lock_t *memory_lock;
#define MEMORY_LOCK_ID 0

// Queue for inter-core commands
queue_t core0_to_core1_queue;
queue_t core1_to_core0_queue;

// Commands for inter-core communication
enum CoreCommand {
    CMD_PROCESS_IMAGE = 1,
    CMD_INFERENCE_COMPLETE = 2,
    CMD_ERROR = 3
};

// Shared data structure for inference results
struct InferenceResult {
    float scores[4];
    uint8_t predictions[4];
    uint32_t inference_time_ms;
    bool valid;
};

// Pre-allocated buffers to reduce memory fragmentation
// Buffer for camera capture - shared between cores
constexpr int kTensorArenaSize = 170 * 1024;  // 150KB for tensor arena
static uint8_t g_capture_buffer[16384] __attribute__((aligned(8)));
static uint32_t g_capture_size = 0;

// Buffer for image processing - used by core 1 - RGB buffer.
// Allocate generous maximum (supports up to 128x128x3). Update if deploying larger models.
static constexpr int kMaxInputWidth  = 128;
static constexpr int kMaxInputHeight = 128;
static uint8_t g_process_buffer[kMaxInputWidth * kMaxInputHeight * 3] __attribute__((aligned(8)));

// Runtime model input dimensions (populated after interpreter initialization on Core 1)
static volatile int g_model_input_width  = 96;
static volatile int g_model_input_height = 96;
static volatile int g_model_input_channels = 3;

// Results shared between cores
static volatile InferenceResult g_inference_result __attribute__((aligned(8)));

// Thresholds and class names sourced from generated metadata
static constexpr float g_improved_thresholds[4] = { 0.55000001f, 0.30000001f, 0.34999999f, 0.25f };
static constexpr const char* g_class_names[4] = { "person", "dog", "cat", "none" };

// Forward declarations
bool setup_hardware();
bool setup_camera();
bool capture_image_to_buffer(uint8_t *buffer, size_t buffer_size, uint32_t *captured_size);
void debug_print(const char *msg);
void print_inference_results(const InferenceResult* result);
static void postprocess_output(const TfLiteTensor* output, InferenceResult* result); // new helper

// TensorFlow Lite globals for Core 1
namespace {
    tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter *error_reporter = &micro_error_reporter;
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;
    alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
}

// Helper function for debug output
void debug_print(const char *msg) {
#if DEBUG
    printf("%s\n", msg);
#endif
}

// Function to print inference results
void print_inference_results(const InferenceResult* result) {
    if (!result->valid) {
        printf("No valid results\n");
        return;
    }
    
    printf("\n=== Detection Results (Time: %lu ms) ===\n", result->inference_time_ms);
    
    bool detected_anything = false;
    for (int i = 0; i < 4; i++) {
        if (result->predictions[i]) {
            printf("  %s: %.3f (DETECTED)\n", g_class_names[i], result->scores[i]);
            detected_anything = true;
        } else {
            printf("  %s: %.3f\n", g_class_names[i], result->scores[i]);
        }
    }
    
    if (!detected_anything) {
        printf("  No objects detected above threshold\n");
    }
    
    printf("==========================================\n\n");
}

// Post-process model output into scores & binary decisions (multi-label)
// Supports uint8, int8, or float output tensors. Applies per-class thresholds.
static void postprocess_output(const TfLiteTensor* out, InferenceResult* result) {
    if (!out || !result) return;
    if (out->type == kTfLiteUInt8) {
        const uint8_t *data = out->data.uint8;
        for (int i = 0; i < 4; ++i) {
            float score = data[i] / 255.0f;
            result->scores[i] = score;
            result->predictions[i] = (score > g_improved_thresholds[i]);
        }
    } else if (out->type == kTfLiteInt8) {
        const int8_t *data = out->data.int8;
        float scale = out->params.scale;
        int zp = out->params.zero_point;
        for (int i = 0; i < 4; ++i) {
            float score = scale * (data[i] - zp);
            if (score < 0.f) score = 0.f; else if (score > 1.f) score = 1.f;
            result->scores[i] = score;
            result->predictions[i] = (score > g_improved_thresholds[i]);
        }
    } else if (out->type == kTfLiteFloat32) {
        const float *data = out->data.f;
        for (int i = 0; i < 4; ++i) {
            float score = data[i];
            if (score < 0.f) score = 0.f; else if (score > 1.f) score = 1.f;
            result->scores[i] = score;
            result->predictions[i] = (score > g_improved_thresholds[i]);
        }
    } else {
        for (int i = 0; i < 4; ++i) {
            result->scores[i] = 0.f;
            result->predictions[i] = 0;
        }
    }
    result->valid = true;
}

// Initialize hardware components
bool setup_hardware() {
    // Initialize stdio
    stdio_init_all();
    sleep_ms(1000); // Give USB time to initialize

    // Set CPU clock to maximum
    set_sys_clock_khz(133000, true);

    debug_print("Initializing hardware...");

    // Initialize SPI for camera communication
    spi_init(SPI_PORT, 8000000); // 8 MHz SPI clock
    gpio_set_function(PIN_SCK, GPIO_FUNC_SPI);
    gpio_set_function(PIN_COPI, GPIO_FUNC_SPI);
    gpio_set_function(PIN_CIPO, GPIO_FUNC_SPI);

    // Initialize CS pin
    gpio_init(PIN_CS);
    gpio_set_dir(PIN_CS, GPIO_OUT);
    gpio_put(PIN_CS, 1);
    
    // Initialize spinlock for memory protection
    memory_lock = spin_lock_init(MEMORY_LOCK_ID);
    
    // Initialize queues for inter-core communication
    queue_init(&core0_to_core1_queue, sizeof(uint32_t), 4);
    queue_init(&core1_to_core0_queue, sizeof(uint32_t), 4);
    
    debug_print("Hardware initialized");
    return true;
}

bool setup_camera() {
    debug_print("Setting up camera...");

    myCAM.begin();
    debug_print("Camera initialized");
    
    // Enable all auto settings
    myCAM.setRotation(CAM_ROTATION_180);
    debug_print("Camera rotation set to 180 degrees");
    myCAM.setAutoFocus(1);
    debug_print("Camera autofocus enabled");
    myCAM.setAutoExposure(1);
    debug_print("Camera auto exposure enabled");
    myCAM.setAutoWhiteBalance(1);
    debug_print("Camera auto white balance enabled");
    myCAM.setAutoISOSensitive(1);
    debug_print("Camera auto ISO enabled");

    debug_print("Camera setup complete");
    return true;
}

// Process image for inference - runs on Core 1
bool process_image_for_inference(const uint8_t *raw_buffer, uint32_t raw_size, uint8_t *output_buffer,
                                 int target_w, int target_h) {
    // Decode JPEG to RGB at the model's resolution
    uint32_t decoded_width, decoded_height;
    
    // First decode to RGB
    if (!jpeg_decode_to_rgb(raw_buffer, raw_size, output_buffer, &decoded_width, &decoded_height)) {
        printf("Core 1: Error - JPEG decoding failed\n");
        return false;
    }
    
    printf("Core 1: Decoded JPEG %lux%lu to RGB\n", decoded_width, decoded_height);
    
    // If the decoded size doesn't match model input, we need to resize
    if (decoded_width != (uint32_t)target_w || decoded_height != (uint32_t)target_h) {
        printf("Core 1: Warning - Image size mismatch. Decoded: %lux%lu, Expected: %dx%d\n",
               decoded_width, decoded_height, target_w, target_h);

        if (target_w > kMaxInputWidth || target_h > kMaxInputHeight) {
            printf("Core 1: ERROR - target dims exceed buffer (%dx%d vs max %dx%d)\n",
                   target_w, target_h, kMaxInputWidth, kMaxInputHeight);
            return false;
        }

        // Simple nearest neighbor resize using stack allocation for temp buffer
        uint8_t *temp_buffer = (uint8_t *)alloca((size_t)target_w * (size_t)target_h * 3);
        for (int y = 0; y < target_h; y++) {
            for (int x = 0; x < target_w; x++) {
                int src_x = (x * (int)decoded_width) / target_w;
                int src_y = (y * (int)decoded_height) / target_h;
                if (src_x >= (int)decoded_width)  src_x = decoded_width - 1;
                if (src_y >= (int)decoded_height) src_y = decoded_height - 1;
                int src_idx = (src_y * decoded_width + src_x) * 3;
                int dst_idx = (y * target_w + x) * 3;
                temp_buffer[dst_idx + 0] = output_buffer[src_idx + 0];
                temp_buffer[dst_idx + 1] = output_buffer[src_idx + 1];
                temp_buffer[dst_idx + 2] = output_buffer[src_idx + 2];
            }
        }
        memcpy(output_buffer, temp_buffer, (size_t)target_w * (size_t)target_h * 3);
    }
    
    return true;
}

// Fill TFLite input tensor with preprocessed image data
bool fill_input_tensor(const uint8_t *image_data) {
    if (!input) {
        printf("Error: Input tensor not initialized\n");
        return false;
    }

    int height = input->dims->data[1];
    int width  = input->dims->data[2];
    int channels = input->dims->data[3];
    if (channels != 3) {
        printf("Error: Unsupported channel count %d (expected 3)\n", channels);
        return false;
    }
    g_model_input_width = width;
    g_model_input_height = height;
    g_model_input_channels = channels;

    // The model expects inputs in specific format based on tensor type
    if (input->type == kTfLiteFloat32) {
        float *input_data = input->data.f;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int src_idx = (y * width + x) * 3;
                int dst_idx = (y * width + x) * 3;
                for (int c = 0; c < 3; c++) {
                    input_data[dst_idx + c] = image_data[src_idx + c] / 255.0f;
                }
            }
        }
    } else if (input->type == kTfLiteInt8) {
        int8_t *input_data = input->data.int8;
        // Int8 tensors usually have specific quantization parameters
        float scale = input->params.scale;
        int zero_point = input->params.zero_point;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int src_idx = (y * width + x) * 3;
                int dst_idx = (y * width + x) * 3;
                for (int c = 0; c < 3; c++) {
                    // Convert to float and then quantize to int8
                    float pixel_value = image_data[src_idx + c] / 255.0f;
                    input_data[dst_idx + c] = (int8_t)((pixel_value / scale) + zero_point);
                }
            }
        }
    } else if (input->type == kTfLiteUInt8) {
        uint8_t *input_data = input->data.uint8;
        // UInt8 tensors might have specific quantization parameters
        float scale = input->params.scale;
        int zero_point = input->params.zero_point;
        
        if (scale == 0) {
            // No quantization, just copy the data
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int src_idx = (y * width + x) * 3;
                    int dst_idx = (y * width + x) * 3;
                    for (int c = 0; c < 3; c++) {
                        input_data[dst_idx + c] = image_data[src_idx + c];
                    }
                }
            }
        } else {
            // Apply quantization
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int src_idx = (y * width + x) * 3;
                    int dst_idx = (y * width + x) * 3;
                    for (int c = 0; c < 3; c++) {
                        float pixel_value = image_data[src_idx + c] / 255.0f;
                        input_data[dst_idx + c] = (uint8_t)((pixel_value / scale) + zero_point);
                    }
                }
            }
        }
    } else {
        printf("Error: Unsupported input tensor type: %d\n", input->type);
        return false;
    }
    return true;
}

// Core 1 entry function - handles all ML inference
void core1_entry() {
    printf("Core 1: Starting TensorFlow Lite initialization...\n");
    
    // Basic sanity check: model_data must not be null & length plausible.
    if (model_data_len == 0 || model_data == nullptr) {
        printf("Core 1: ERROR - model data not present\n");
        uint32_t error_cmd = CMD_ERROR;
        queue_try_add(&core1_to_core0_queue, &error_cmd);
        return;
    }
    
    // Use the model data from the header file
    model = tflite::GetModel(model_data);
    if (!model) {
        printf("Core 1: ERROR - Failed to get TFLite model\n");
        uint32_t error_cmd = CMD_ERROR;
        queue_try_add(&core1_to_core0_queue, &error_cmd);
        return;
    }
    
    // Create a resolver with the operations needed for the model
    static tflite::MicroMutableOpResolver<16> resolver;
    
    // Core operations for MobileNet models
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddPad();
    resolver.AddConcatenation();
    resolver.AddRelu6();
    resolver.AddLogistic();
    
    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("Core 1: ERROR - Failed to allocate tensors: %d\n", allocate_status);
        uint32_t error_cmd = CMD_ERROR;
        queue_try_add(&core1_to_core0_queue, &error_cmd);
        return;
    }
    
    // Get input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // Cache initial input dims (expect 4D: [1, H, W, C])
    if (input && input->dims && input->dims->size == 4) {
     g_model_input_height = input->dims->data[1];
     g_model_input_width  = input->dims->data[2];
     g_model_input_channels = input->dims->data[3];
    }
    printf("Core 1: TensorFlow Lite initialized successfully\n");
    printf("Core 1: Model size: %lu bytes\n", model_data_len);
    printf("Core 1: Input tensor type: %d, dims: %d x %d x %d x %d\n",
        input->type, input->dims->data[0], input->dims->data[1],
        input->dims->data[2], input->dims->data[3]);
    printf("Core 1: Output tensor type: %d, dims: %d x %d\n",
           output->type, output->dims->data[0], output->dims->data[1]);
    printf("Core 1: Cached input dims: %dx%dx%d\n", g_model_input_width, g_model_input_height, g_model_input_channels);
    
    // Signal that initialization is complete
    uint32_t init_complete = CMD_INFERENCE_COMPLETE;
    queue_try_add(&core1_to_core0_queue, &init_complete);
    
    // Main processing loop
    uint32_t command;
    
    while (true) {
        // Wait for a command from Core 0
        if (queue_try_remove(&core0_to_core1_queue, &command)) {
            if (command == CMD_PROCESS_IMAGE) {
                // Get a lock on the shared memory
                uint32_t save = spin_lock_blocking(memory_lock);
                
                // Process the captured image
                bool process_success = process_image_for_inference(
                    g_capture_buffer, g_capture_size, g_process_buffer,
                    g_model_input_width, g_model_input_height);
                
                // Release the lock
                spin_unlock(memory_lock, save);
                
                if (!process_success) {
                    printf("Core 1: Image processing failed\n");
                    uint32_t error_cmd = CMD_ERROR;
                    queue_try_add(&core1_to_core0_queue, &error_cmd);
                    continue;
                }
                
                // Fill input tensor with processed image
                fill_input_tensor(g_process_buffer);
                
                // Run inference with timing
                absolute_time_t inference_start = get_absolute_time();
                
                TfLiteStatus invoke_status = interpreter->Invoke();
                
                uint32_t inference_time_ms = absolute_time_diff_us(
                    inference_start, get_absolute_time()) / 1000;
                
          printf("Core 1: Inference took %lu ms (input %dx%dx%d)\n", inference_time_ms,
              g_model_input_width, g_model_input_height, g_model_input_channels);
                
                if (invoke_status != kTfLiteOk) {
                    printf("Core 1: Inference failed with status: %d\n", invoke_status);
                    uint32_t error_cmd = CMD_ERROR;
                    queue_try_add(&core1_to_core0_queue, &error_cmd);
                    continue;
                }
                
                // Process results and store in shared memory
                save = spin_lock_blocking(memory_lock);
                InferenceResult* result = (InferenceResult*)&g_inference_result;
                result->inference_time_ms = inference_time_ms;
                postprocess_output(output, result); // use helper
                spin_unlock(memory_lock, save);
                
                // Signal that inference is complete
                uint32_t complete_cmd = CMD_INFERENCE_COMPLETE;
                queue_try_add(&core1_to_core0_queue, &complete_cmd);
            }
        }
        // Small sleep to avoid tight loop
        sleep_us(100);
    }
}

// Capture a JPEG image from Arducam into buffer
bool capture_image_to_buffer(uint8_t *buffer, size_t buffer_size, uint32_t *captured_size) {
    debug_print("Capturing image from Arducam...");

    // Take a picture in 128x128 JPEG mode
    CamStatus status = myCAM.takePicture(CAM_IMAGE_MODE_128X128, CAM_IMAGE_PIX_FMT_JPG);
    if (status != CAM_ERR_SUCCESS) {
        printf("Arducam: takePicture failed (%d)\n", status);
        return false;
    }

    // Get the image length
    uint32_t img_len = myCAM.getTotalLength();
    if (img_len == 0 || img_len > buffer_size) {
        printf("Arducam: Invalid image length: %lu\n", img_len);
        return false;
    }

    // Read image data into buffer in chunks (max 255 bytes per call)
    uint32_t bytes_read = 0;
    while (bytes_read < img_len) {
        uint8_t chunk = 128;
        if (img_len - bytes_read < chunk) chunk = img_len - bytes_read;
        if (chunk > 255) chunk = 255;
        bytes_read += myCAM.readBuff(buffer + bytes_read, chunk);
    }

    *captured_size = img_len;
    printf("Arducam: Image captured, size: %lu bytes\n", img_len);
    return true;
}

// Print memory information
void print_memory_info() {
    extern char __StackLimit, __bss_end__;
    printf("Free RAM: %d bytes\n", &__StackLimit - &__bss_end__);
}

// Main function - runs on Core 0
int main() {
    // Set inference result as invalid initially
    g_inference_result.valid = false;

    // Initialize all components
    if (!setup_hardware()) {
        printf("Hardware setup failed\n");
        while (true)
            sleep_ms(1000); // Halt
    }
    
    if (!setup_camera()) {
        printf("Camera setup failed\n");
        while (true)
            sleep_ms(1000); // Halt
    }
    
    print_memory_info();
    
    // Launch Core 1 for ML processing
    multicore_launch_core1(core1_entry);
    
    // Wait for Core 1 to initialize TFLite
    printf("Core 0: Waiting for TensorFlow Lite initialization...\n");
    
    uint32_t response;
    bool init_success = false;
    absolute_time_t timeout = make_timeout_time_ms(10000);  // 10 second timeout
    
    while (!init_success) {
        if (queue_try_remove(&core1_to_core0_queue, &response)) {
            if (response == CMD_INFERENCE_COMPLETE) {
                init_success = true;
            }
            else if (response == CMD_ERROR) {
                printf("Core 0: Error during TensorFlow initialization\n");
                while (true)
                    sleep_ms(1000); // Halt
            }
        }
        
        // Check for timeout
        if (absolute_time_diff_us(get_absolute_time(), timeout) <= 0) {
            printf("Core 0: Timeout waiting for TensorFlow initialization\n");
            while (true)
                sleep_ms(1000); // Halt
        }
        
        sleep_ms(10);
    }
    
    printf("Core 0: TensorFlow Lite initialization complete!\n");
    printf("Core 0: Starting main detection loop...\n");
    
    // Main processing loop
    while (true) {
        printf("Core 0: Capturing new image...\n");
        
        // Capture image
        uint32_t capture_size = 0;
        bool capture_success = capture_image_to_buffer(
            g_capture_buffer, sizeof(g_capture_buffer), &capture_size);
            
        if (!capture_success) {
            printf("Core 0: Image capture failed\n");
            sleep_ms(1000);
            continue;
        }
        
        // Set the capture size in the shared memory
        uint32_t save = spin_lock_blocking(memory_lock);
        g_capture_size = capture_size;
        spin_unlock(memory_lock, save);
        
        printf("Core 0: Processing image...\n");
        
        // Signal Core 1 to process the image
        uint32_t process_cmd = CMD_PROCESS_IMAGE;
        queue_try_add(&core0_to_core1_queue, &process_cmd);
        
        // Wait for Core 1 to complete processing
        bool processing_complete = false;
        bool processing_error = false;
        absolute_time_t timeout = make_timeout_time_ms(5000);  // 5 second timeout
        
        while (!processing_complete && !processing_error) {
            // Check for response from Core 1
            uint32_t response;
            if (queue_try_remove(&core1_to_core0_queue, &response)) {
                if (response == CMD_INFERENCE_COMPLETE) {
                    processing_complete = true;
                }
                else if (response == CMD_ERROR) {
                    processing_error = true;
                    printf("Core 0: Error during inference\n");
                }
            }
            
            // Check for timeout
            if (absolute_time_diff_us(get_absolute_time(), timeout) <= 0) {
                printf("Core 0: Timeout waiting for inference\n");
                processing_error = true;
            }
            
            sleep_ms(10);
        }
        
        if (processing_error) {
            printf("Core 0: Processing error, retrying...\n");
            sleep_ms(1000);
            continue;
        }
        
        // Print the inference results
        save = spin_lock_blocking(memory_lock);
        InferenceResult result;
        // Manually copy volatile struct members
        for (int i = 0; i < 4; i++) {
            result.scores[i] = g_inference_result.scores[i];
            result.predictions[i] = g_inference_result.predictions[i];
        }
        result.inference_time_ms = g_inference_result.inference_time_ms;
        result.valid = g_inference_result.valid;
        spin_unlock(memory_lock, save);
        
        print_inference_results(&result);
               
        // Delay before next capture
        sleep_ms(2000); // 2 second delay between detections
    }

    return 0;
}