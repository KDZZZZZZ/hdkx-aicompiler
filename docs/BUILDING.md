# Building hdkx-aicompiler

This project supports a CPU-only development path that does not require CUDA,
NCCL, or a GPU. On Windows, the most reliable local path in a plain PowerShell
session is the MinGW preset.

## Windows: MinGW CPU-only

Prerequisites:

- CMake 3.20 or newer
- Ninja
- MinGW `g++` on `PATH`

Check the tools:

```powershell
where.exe cmake
where.exe ninja
where.exe g++
```

Configure and build:

```powershell
cmake --preset dev-mingw-cpu
cmake --build --preset dev-mingw-cpu
```

Run CPU smoke targets:

```powershell
cmake --build out/build/dev-mingw-cpu --target run_pass_pipeline_test
cmake --build out/build/dev-mingw-cpu --target run_profile_bundle_test
```

The `dev-mingw-cpu` preset disables CUDA and LLVM. This keeps the build focused
on the portable runtime, Relay/TIR pass tests, profiling smoke test, and IR dump
tools.

## Windows: MSVC CPU-only

The `dev-ninja-cpu` preset uses the default CMake compiler discovery. If CMake
selects MSVC, the shell must have the full Visual Studio C++ and Windows SDK
tools available.

Required tools include:

- `cl.exe`
- `link.exe`
- `rc.exe`
- `mt.exe`

Check the tools:

```powershell
where.exe cl
where.exe rc
where.exe mt
```

If configure fails with `CMAKE_MT-NOTFOUND` or a manifest/resource error, open
an x64 Native Tools Command Prompt/PowerShell for Visual Studio, or install the
Windows SDK component that provides `rc.exe` and `mt.exe`.

Configure and build:

```powershell
cmake --preset dev-ninja-cpu
cmake --build --preset dev-ninja-cpu
```

## Existing Presets

| Preset | Purpose |
|---|---|
| `dev-ninja` | Ninja Debug build with CUDA detection enabled |
| `dev-ninja-cpu` | Ninja Debug build with CUDA disabled; uses default compiler discovery |
| `dev-mingw-cpu` | Ninja Debug build with MinGW `g++`, CUDA disabled, LLVM disabled |
| `dev-ninja-debug-examples` | Ninja Debug build with extra temporary debug examples |
