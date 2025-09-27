// Thin translation unit to provide model symbols expected by the build system.
// The actual binary array is defined in model_data.h (auto-generated).
// Having a .cc file allows CMake target list (src/model_data.cc) to succeed
// without duplicating the large array or increasing flash usage.

#include "model_data.h"

// Nothing else needed.
