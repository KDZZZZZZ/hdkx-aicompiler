/*! \file src/profiling/options.cc
 * \brief Implements profiling configuration and stable formatting helpers.
 */

#include "kxc/profiling/profiling.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>

namespace kxc::profiling {
namespace {

std::string GetEnvString(const char* key) {
  const char* value = std::getenv(key);
  return value == nullptr ? "" : value;
}

bool ParseEnvBool(const std::string& value, bool default_value) {
  if (value.empty()) return default_value;
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);
  return lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on";
}

}  // namespace

ProfileOptions ApplyEnvironmentOverrides(ProfileOptions options) {
  options.enabled = ParseEnvBool(GetEnvString("KXC_PROFILE_ENABLE"), options.enabled);
  const std::string bundle_dir = GetEnvString("KXC_PROFILE_BUNDLE_DIR");
  if (!bundle_dir.empty()) options.bundle_dir = bundle_dir;

  const std::string log_level = GetEnvString("KXC_PROFILE_LOG_LEVEL");
  if (!log_level.empty()) {
    std::string lowered = log_level;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);
    if (lowered == "debug") options.log_level = LogSeverity::kDebug;
    if (lowered == "info") options.log_level = LogSeverity::kInfo;
    if (lowered == "warn" || lowered == "warning") options.log_level = LogSeverity::kWarn;
    if (lowered == "error") options.log_level = LogSeverity::kError;
  }

  const std::string ir_mode = GetEnvString("KXC_PROFILE_IR_MODE");
  if (!ir_mode.empty()) {
    std::string lowered = ir_mode;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);
    if (lowered == "disabled" || lowered == "off") {
      options.ir_capture_mode = IRCaptureMode::kDisabled;
    } else if (lowered == "verbose" || lowered == "all") {
      options.ir_capture_mode = IRCaptureMode::kVerbose;
    } else {
      options.ir_capture_mode = IRCaptureMode::kChangedOrFailed;
    }
  }

  options.enable_nvtx = ParseEnvBool(GetEnvString("KXC_PROFILE_NVTX"), options.enable_nvtx);
  options.enable_cupti = ParseEnvBool(GetEnvString("KXC_PROFILE_CUPTI"), options.enable_cupti);
  options.record_pass_ir =
      ParseEnvBool(GetEnvString("KXC_PROFILE_RECORD_PASS_IR"), options.record_pass_ir);
  options.record_execution_plan_details =
      ParseEnvBool(GetEnvString("KXC_PROFILE_EXEC_PLAN_DETAILS"),
                   options.record_execution_plan_details);
  return options;
}

std::string LogSeverityToString(LogSeverity severity) {
  switch (severity) {
    case LogSeverity::kDebug: return "debug";
    case LogSeverity::kInfo: return "info";
    case LogSeverity::kWarn: return "warn";
    case LogSeverity::kError: return "error";
  }
  return "info";
}

std::string IRCaptureModeToString(IRCaptureMode mode) {
  switch (mode) {
    case IRCaptureMode::kDisabled: return "disabled";
    case IRCaptureMode::kChangedOrFailed: return "changed_or_failed";
    case IRCaptureMode::kVerbose: return "verbose";
  }
  return "changed_or_failed";
}

std::string ShapeSignatureToString(const std::vector<std::vector<int64_t>>& shapes) {
  std::ostringstream os;
  os << "[";
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    if (i) os << ",";
    os << "[";
    for (std::size_t j = 0; j < shapes[i].size(); ++j) {
      if (j) os << ",";
      os << shapes[i][j];
    }
    os << "]";
  }
  os << "]";
  return os.str();
}

StringMap MakeFields(std::initializer_list<std::pair<std::string, std::string>> init) {
  StringMap fields;
  for (const auto& item : init) fields[item.first] = item.second;
  return fields;
}

bool ShouldCaptureIR(const std::shared_ptr<ProfileContext>& ctx, bool changed, bool failed) {
  if (!ctx || !ctx->options().record_pass_ir) return false;
  switch (ctx->options().ir_capture_mode) {
    case IRCaptureMode::kDisabled:
      return false;
    case IRCaptureMode::kVerbose:
      return true;
    case IRCaptureMode::kChangedOrFailed:
      return changed || failed;
  }
  return changed || failed;
}

}  // namespace kxc::profiling
