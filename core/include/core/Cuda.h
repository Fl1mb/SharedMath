#pragma once

/// Convenience umbrella header for the CUDA management subsystem.
///
///   #include "core/Cuda.h"
///
///   auto& disp = SharedMath::Core::CudaDispatcher::instance();
///   auto  fut  = disp.submit([](int dev) { /* your kernel */ return 42; });
///   int   val  = fut.get();

#include "CudaDeviceInfo.h"
#include "CudaDeviceManager.h"
#include "CudaDispatcher.h"
