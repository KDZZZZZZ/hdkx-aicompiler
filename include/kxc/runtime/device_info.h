/*! \file include/kxc/runtime/device_info.h
 * \brief Defines backend-independent device capability data.
 */

#pragma once

#include <cstdint>
#include <string>

#include "kxc/runtime/device.h"

namespace kxc {

struct DeviceAttributes {
    int exists{0};
    int64_t max_threads_per_block{1};
    int64_t warp_size{1};
    int64_t max_shared_memory_per_block{0};
    std::string compute_version{"0.0"};
    std::string device_name;
    int64_t max_clock_rate_khz{0};
    int64_t max_registers_per_block{0};
    int64_t api_version{0};
    int64_t driver_version{0};
    int64_t l2_cache_size_bytes{0};
    int64_t total_global_memory{0};
    int64_t available_global_memory{0};
    int64_t max_shared_memory_per_multiprocessor{0};
    int64_t max_registers_per_multiprocessor{0};
    int64_t max_threads_per_multiprocessor{0};
    int64_t compute_version_major{0};
    int64_t compute_version_minor{0};
    int64_t multi_processor_count{1};
    std::string arch;
};

struct DeviceInfo {
    DeviceTypeCode device_type{kUnknown};
    int device_id{-1};
    std::string device_type_name;
    std::string target_kind;
    bool available{false};
    std::string status;
    DeviceAttributes attrs;
};

}  // namespace kxc
