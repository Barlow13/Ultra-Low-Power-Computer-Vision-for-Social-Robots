// Auto-generated from TensorFlow/export/metadata.json
// Contains model class labels and per-class thresholds.
// Model: SmallMobileNetV2 (int8) 96x96x3
// Generated: 2025-09-27T01:31:55.169067
// Update procedure: copy new metadata.json and regenerate this header.

#ifndef MODEL_METADATA_H
#define MODEL_METADATA_H

#include <cstdint>

namespace model_metadata {
constexpr int kNumClasses = 4;
constexpr const char* kClassNames[kNumClasses] = {"person", "dog", "cat", "none"};
constexpr float kThresholds[kNumClasses] = {0.55000001f, 0.30000001f, 0.34999999f, 0.25f};
// Input shape (H, W, C)
constexpr int kInputHeight = 96;
constexpr int kInputWidth  = 96;
constexpr int kInputChannels = 3;
}

#endif // MODEL_METADATA_H
