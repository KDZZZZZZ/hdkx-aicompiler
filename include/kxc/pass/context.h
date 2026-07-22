/*! \file include/kxc/pass/context.h
 * \brief Defines the IR-independent pass execution context.
 */

#pragma once

#include <memory>
#include <string>

#include "kxc/target/virtual_device.h"

namespace kxc {

class PassContext {
public:
    PassContext() = default;

    bool defined() const { return defined_; }
    bool is_multi_device() const { return is_multi_device_; }
    const VirtualDevice& primary_virtual_device() const {
        return primary_virtual_device_;
    }
    const Array<VirtualDevice>& virtual_devices() const {
        return virtual_devices_;
    }
    const Target& default_target() const { return default_target_; }
    const Device& default_device() const { return default_device_; }

    std::string ToString() const;

    static PassContext Current();
    static PassContext FromVirtualDevices(
        const Array<VirtualDevice>& virtual_devices);
    static PassContext FromComponents(Array<VirtualDevice> virtual_devices,
                                      Target default_target,
                                      Device default_device,
                                      bool is_multi_device,
                                      bool defined);
    static PassContext FromTarget(const Target& target);
    static PassContext MergeTarget(const PassContext& base_ctx,
                                   const Target& target);

    class Scope {
    public:
        explicit Scope(const PassContext& pass_ctx);
        ~Scope();

        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        bool active_{false};
        std::unique_ptr<PassContext> previous_;
    };

private:
    bool defined_{false};
    bool is_multi_device_{false};
    VirtualDevice primary_virtual_device_;
    Array<VirtualDevice> virtual_devices_;
    Target default_target_;
    Device default_device_;

    static void SetCurrent(const PassContext& pass_ctx);
    static void ClearCurrent();
};

}  // namespace kxc
