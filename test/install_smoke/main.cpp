#include <kxc/ffi/registry.h>
#include <kxc/relay/op.h>
#include <kxc/runtime/device.h>

int main() {
  const kxc::Device device = kxc::Device::CPU(0);
  const kxc::relay::Op& add = kxc::relay::Op::Get("add");
  const kxc::PackedFunc make_add =
      kxc::Registry::Global().Get("kxc.relay.op._make.add");
  const bool valid = device.device_type() == kxc::kCPU &&
                     device.device_id() == 0 && add->num_inputs == 2 &&
                     static_cast<bool>(make_add);
  return valid ? 0 : 1;
}
