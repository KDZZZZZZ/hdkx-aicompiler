/*! \file src/pass/context.cc
 * \brief Implements the IR-independent pass execution context.
 */

#include "kxc/pass/context.h"

#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace kxc {
namespace {

thread_local PassContext current_pass_ctx;
thread_local bool has_current_pass_ctx = false;

bool IsTargetObjectRef(const ObjectRef& obj) {
    return obj.defined() && obj.get()->GetTypeId() == TargetNode::_type_index;
}

std::string DeviceIdentityKey(const Device& device, const Target& target) {
    if (device.defined()) {
        return std::to_string(static_cast<int>(device.device_type())) + ":" +
               std::to_string(device.device_id());
    }
    if (target.defined()) {
        return std::to_string(static_cast<int>(target->device_type)) + ":" +
               std::to_string(target->device_id);
    }
    return "unconstrained";
}

}  // namespace

std::string PassContext::ToString() const {
    if (!defined_) return "PassContext(undefined)";

    std::stringstream ss;
    ss << "PassContext(multi_device=" << (is_multi_device_ ? "true" : "false")
       << ", virtual_devices=" << virtual_devices_.size();
    if (primary_virtual_device_.defined()) {
        ss << ", primary=" << primary_virtual_device_.ToString();
    } else {
        ss << ", primary=none";
    }
    if (default_target_.defined()) {
        ss << ", target=" << default_target_.ToString();
    } else {
        ss << ", target=none";
    }
    ss << ")";
    return ss.str();
}

PassContext PassContext::Current() {
    return has_current_pass_ctx ? current_pass_ctx : PassContext();
}

PassContext PassContext::FromVirtualDevices(
    const Array<VirtualDevice>& virtual_devices) {
    if (virtual_devices.empty()) return PassContext();

    PassContext ctx;
    ctx.defined_ = true;
    ctx.virtual_devices_ = virtual_devices;
    ctx.primary_virtual_device_ = virtual_devices[0];
    ctx.default_device_ = ctx.primary_virtual_device_->device;
    if (ctx.primary_virtual_device_->target.defined()) {
        ctx.default_target_ = ctx.primary_virtual_device_->target;
    } else if (ctx.default_device_.defined()) {
        ctx.default_target_ = BuildTarget(ctx.default_device_);
    }

    std::unordered_set<std::string> identities;
    for (const auto& vd : virtual_devices) {
        if (vd.defined()) {
            identities.insert(DeviceIdentityKey(vd->device, vd->target));
        }
    }
    ctx.is_multi_device_ = identities.size() > 1;
    return ctx;
}

PassContext PassContext::FromComponents(
    Array<VirtualDevice> virtual_devices, Target default_target,
    Device default_device, bool is_multi_device, bool defined) {
    PassContext ctx;
    ctx.defined_ = defined;
    ctx.virtual_devices_ = std::move(virtual_devices);
    if (!ctx.virtual_devices_.empty()) {
        ctx.primary_virtual_device_ = ctx.virtual_devices_[0];
    }
    ctx.default_target_ = std::move(default_target);
    ctx.default_device_ = std::move(default_device);
    ctx.is_multi_device_ = is_multi_device;
    return ctx;
}

PassContext PassContext::FromTarget(const Target& target) {
    if (!IsTargetObjectRef(target) || target->device_type == kUnknown ||
        target->device_id < 0 || target->kind.empty()) {
        throw std::invalid_argument("PassContext requires a complete Target");
    }
    const Device device(target->device_type, target->device_id);
    return FromVirtualDevices(
        {VirtualDevice::ForDeviceAndTarget(device, target)});
}

PassContext PassContext::MergeTarget(const PassContext& base_ctx,
                                     const Target& target) {
    PassContext target_ctx = FromTarget(target);
    if (!base_ctx.defined()) return target_ctx;

    const Device expected(target->device_type, target->device_id);
    if (base_ctx.default_device_.defined() &&
        base_ctx.default_device_ != expected) {
        throw std::invalid_argument(
            "Relay placement device conflicts with CompileConfig target");
    }
    if (base_ctx.default_target_.defined()) {
        const Target& placed = base_ctx.default_target_;
        if (!IsTargetObjectRef(placed) || placed->kind != target->kind ||
            placed->device_type != target->device_type ||
            placed->device_id != target->device_id) {
            throw std::invalid_argument(
                "Relay placement target conflicts with CompileConfig target");
        }
    }

    PassContext merged = base_ctx;
    merged.defined_ = true;
    merged.default_target_ = target;
    merged.default_device_ = expected;
    if (!merged.primary_virtual_device_.defined()) {
        merged.primary_virtual_device_ =
            VirtualDevice::ForDeviceAndTarget(expected, target);
        merged.virtual_devices_ = {merged.primary_virtual_device_};
    }
    return merged;
}

void PassContext::SetCurrent(const PassContext& pass_ctx) {
    current_pass_ctx = pass_ctx;
    has_current_pass_ctx = pass_ctx.defined_;
}

void PassContext::ClearCurrent() {
    current_pass_ctx = PassContext();
    has_current_pass_ctx = false;
}

PassContext::Scope::Scope(const PassContext& pass_ctx) {
    previous_ = std::make_unique<PassContext>(PassContext::Current());
    if (!pass_ctx.defined()) return;
    PassContext::SetCurrent(pass_ctx);
    active_ = true;
}

PassContext::Scope::~Scope() {
    if (!active_) return;
    if (previous_ && previous_->defined()) {
        PassContext::SetCurrent(*previous_);
    } else {
        PassContext::ClearCurrent();
    }
}

}  // namespace kxc
