/*! \file src/ffi/builtin_registry.cc
 * \brief Keeps registration-only translation units alive in static-library consumers.
 */

#include "kxc/ffi/registration.h"

namespace kxc::builtin_anchor {

void CompilerDistributed();
void DistributedCclCpu();
void DistributedSession();
void DistributedWorker();
void RelayOpFfi();
void RelayActivationOps();
void RelayConvolutionOps();
void RelayDenseOps();
void RelayLayerNormOps();
void RelayPoolingOps();
void RelaySoftmaxOps();
void RelayTensorMathOps();
void RelayTensorReduceOps();
void RelayTensorTransformOps();
void RelayPasses();
void RuntimeDeviceApi();
void TargetVirtualDevice();
void TirPasses();

}  // namespace kxc::builtin_anchor

namespace kxc {

void RegisterBuiltins() {
  builtin_anchor::CompilerDistributed();
  builtin_anchor::DistributedCclCpu();
  builtin_anchor::DistributedSession();
  builtin_anchor::DistributedWorker();
  builtin_anchor::RelayOpFfi();
  builtin_anchor::RelayActivationOps();
  builtin_anchor::RelayConvolutionOps();
  builtin_anchor::RelayDenseOps();
  builtin_anchor::RelayLayerNormOps();
  builtin_anchor::RelayPoolingOps();
  builtin_anchor::RelaySoftmaxOps();
  builtin_anchor::RelayTensorMathOps();
  builtin_anchor::RelayTensorReduceOps();
  builtin_anchor::RelayTensorTransformOps();
  builtin_anchor::RelayPasses();
  builtin_anchor::RuntimeDeviceApi();
  builtin_anchor::TargetVirtualDevice();
  builtin_anchor::TirPasses();
}

}  // namespace kxc
