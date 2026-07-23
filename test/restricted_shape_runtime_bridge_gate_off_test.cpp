#include <iostream>

#include "kxc/compiler/restricted_shape_runtime_bridge.h"

int main() {
    using kxc::api::experimental::restricted_shape_runtime_bridge::v1::RestrictedShapeRuntimeBridge;
#if KXC_ENABLE_RESTRICTED_SHAPE_RUNTIME_BRIDGE
    if (!RestrictedShapeRuntimeBridge::IsEnabled()) return 1;
#else
    if (RestrictedShapeRuntimeBridge::IsEnabled()) {
        std::cerr << "restricted shape runtime bridge gate-off contract failed\n";
        return 1;
    }
#endif
    return 0;
}
