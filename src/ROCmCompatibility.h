#ifndef _CHANRTS_ROCM_COMPATIBILITY_H
#define _CHANRTS_ROCM_COMPATIBILITY_H

/*
 * ROCm LibTorch 2.7.1 + GCC 13 Compatibility Layer
 * 
 * This header works around fundamental template deduction issues
 * between ROCm LibTorch 2.7.1 and GCC 13's stricter template handling.
 */

#ifdef CHANRTS_ROCM_COMPAT

// Include standard headers first
#include <algorithm>
#include <type_traits>

// Create custom std::min override for the problematic LibTorch usage pattern
namespace std {
    // Specialize for the exact pattern LibTorch uses: min(0.5f, static_cast<double>(...))
    inline constexpr double min(float a, double b) {
        return (a < b) ? static_cast<double>(a) : b;
    }
    
    inline constexpr double min(double a, float b) {
        return (a < static_cast<double>(b)) ? a : static_cast<double>(b);
    }
}

// Workaround for std::format missing in C++17 (used by float3.h)
// Note: We fixed float3.h directly instead of providing std::format implementation
// This section is kept for completeness but not needed with our float3.h fix

// Additional LibTorch compatibility fixes
#define TORCH_DISABLE_ASSERTS 1

// Suppress template warnings that we can't fix
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"

#endif // CHANRTS_ROCM_COMPAT

#endif // _CHANRTS_ROCM_COMPATIBILITY_H