# GPU 驱动一致性修复报告

2026-09-09。磁盘上的驱动安装和启动文件已核验、刷新；当前 Linux 会话仍运行旧内核模块，本机修复生效需要重启。用户随后选择通过 Tailscale 使用 Windows GPU，现已建立 SSH 入口并完成原生 CUDA 专项 **5/5**，见 [Windows 实测报告](GPU_WINDOWS_VALIDATION_REPORT.md)。本机未重启。

## 现象与原因

机器识别到 NVIDIA GeForce GTX 1650，`/dev/nvidia0`、`nvidiactl` 和 `nvidia-uvm` 均存在。`nvidia-smi` 返回 `Driver/library version mismatch`；直接调用 `libcuda.so.1` 的 `cuInit(0)` 返回 804，错误名为 `CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE`。初始化阶段已失败，因此 KXC 的硬件测试报告设备不可用。

| 层 | 实测状态 |
|---|---|
| 当前内核 | 6.17.0-40-generic |
| 已加载 NVIDIA 内核模块 | 580.159.03 |
| 实际加载的 libcuda | `/usr/lib/x86_64-linux-gnu/libcuda.so.580.173.02` |
| 当前内核对应的磁盘模块 | 580.173.02 |
| 默认启动内核 7.0.0-30-generic 的磁盘模块 | 580.173.02 |
| NVIDIA driver、compute、utils、GL、kernel-common 包 | 580.173.02 |

内核日志同时记录 NVRM API mismatch。该组合符合升级用户态驱动后、旧模块仍驻留内核的情况。[Ubuntu 官方处理说明](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/#driver-library-version-mismatch-error)要求重启以加载新模块。804 表示此次兼容路径不受设备支持，具体定义见 [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)；它不能作为显卡缺失的证据。

## 已执行的方法与结果

1. `sudo apt-get check`：包依赖一致，无待修复依赖。
2. `dpkg --verify`：driver、utils、compute 和当前内核的预编译 NVIDIA module 包校验无差异。使用 Ubuntu 的已签名预编译模块，未另装 DKMS 或混用 runfile。
3. 用户要求安装后，执行 `sudo apt-get --no-remove install nvidia-driver-580`：580.173.02 已是安装版本，结果为 0 新装、0 升级、0 删除。没有重复安装已有文件，也未执行 apt 提示的无关 autoremove。
4. 分别执行 `update-initramfs -u -k 6.17.0-40-generic` 和 `update-initramfs -u -k 7.0.0-30-generic`：两份启动映像均刷新成功，退出码 0。
5. 核对 `/boot/vmlinuz` 与 `/boot/initrd.img` 指向 7.0.0-30-generic；GRUB 默认项为 0。当前和默认启动内核的 NVIDIA module 均为 580.173.02。

安装检查日志为 `/tmp/kxc-nvidia-install-check.log`，启动文件生成日志为 `/tmp/kxc-nvidia-initramfs-6.17.0-40-generic.log` 和 `/tmp/kxc-nvidia-initramfs-7.0.0-30-generic.log`。

## 生效与验证边界

Hyprland、Xwayland 和运行中的桌面应用仍持有 NVIDIA 设备。没有强行卸载模块或停止桌面。重启会关闭当前桌面和程序，需要用户确认此中断；磁盘准备工作已经完成。

重启后先检查 `/proc/driver/nvidia/version`、`nvidia-smi` 和 `cuInit(0)`，再执行 `codegen_cuda_test` 的 NVRTC/load/launch 和数值阶段。重启前不能把安装核验或 initramfs 刷新写成 GPU 已可用。

此前 MiniMind-V 视觉阶段的 CPU/LLVM 结果、CUDA 配置 4/5 的实际失败记录保留在 [视觉链技术报告](M9_MINIMIND_V_VISION_REPORT.md)。驱动恢复后再补硬件结果，不追改历史失败。

## Windows GPU 接入与验证

用户指定使用 Tailscale 中的 Windows GPU，并授权配置远程入口。目标为 `BF-202408261826`（`100.121.109.90`）；当前 Linux 的 Tailscale 地址为 `100.77.91.52`。初次 Tailscale ping 约 6 ms，但 SSH 22/2222 和 WinRM 5985/5986 均超时。经过以下配置和认证修复，现已使用 `administrator` 成功远程执行；GPU 为 RTX 4070 Ti SUPER，驱动 576.57。

已生成独立 Ed25519 客户端密钥，私钥保存在 Linux 的 `~/.ssh/id_ed25519_kxc_windows_gpu`，权限 0600。只将公钥嵌入本机部署产物 `out/windows-gpu/kxc-windows-gpu-setup.ps1`。该脚本通过 Windows 自带功能安装 OpenSSH Server，追加限定 Linux 来源地址的公钥，设置管理员公钥文件 ACL，并建立只匹配这两个 Tailscale 地址的 TCP/22 规则。保留原有 SSH 配置、已有公钥和既有防火墙规则；本次安装新建的默认 SSH 规则也收窄地址范围。实现依据为 [Microsoft 安装说明](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse)和[公钥认证说明](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_keymanagement)。

脚本检查 `sshd -t` 和服务实际监听的 TCP/22，随后将用户名、SSH 主机公钥、`nvidia-smi`、WSL 和 nvcc 的检测结果写入 JSON，经 Taildrop 回传。用户已在 Windows 管理员 PowerShell 中执行；首次回执为 `out/windows-gpu/inbox/kxc-windows-gpu-receipt-20260909-124307.json`。

已完成的验证与效果：

1. 客户端公钥与脚本内嵌公钥一致，脚本不含私钥。
2. 使用官方 PowerShell 7.6.6 的解析器检查脚本，语法错误为 0；下载包 SHA256 与官方发布校验文件一致。随后 Windows 执行回执确认 OpenSSH 已监听，客户端核对并固定了回执中的主机公钥。
3. `sudo tailscale file cp` 在 2026-09-09 11:05:31（Asia/Shanghai）返回发送成功。普通用户调用曾被本机 Tailscale 权限拒绝，随后使用现有免密 sudo 完成发送，未修改 Tailscale operator 配置。
4. 按 [Taildrop Windows 接收说明](https://tailscale.com/docs/features/taildrop?tab=windows)，当前版本通常将文件放入用户 Downloads 目录。脚本发送与语法检查只是中间步骤；最终验收依据是 Windows 回执、成功的 SSH 认证和 CUDA 执行。
5. 11:09:01 曾发送 678 份源码快照，包 SHA256 为 `c908813bf5286f3b4fe0c1487a5535dd43f1ff608ccf15e1ad4fd97457ed17f1`。最终构建使用随后更新的 682 份源码包，包含图文联合模块，并核对了全部文件及两份 CUPTI 修复文件的哈希，详见 [Windows 实测报告](GPU_WINDOWS_VALIDATION_REPORT.md)。

初次公钥认证失败时，OpenSSH 日志只记录 preauth 连接关闭。用户随后执行 `kxc-windows-ssh-keyfix.ps1`，回执 `out/windows-gpu/inbox/kxc-windows-keyfix-20260909-125120.json` 给出了有效配置：`authorizedkeysfile .ssh/authorized_keys`。这说明实际使用用户公钥文件，而非初始脚本写入的管理员公共文件。修复脚本备份原内容与 SDDL，在两个位置保留其它公钥并写入本任务限定来源地址的公钥，以无 BOM 编码和正确所有者/ACL 保存；SSH 配置保持原有值。认证现已成功，后续配置均可远程完成。

Windows 的 `cuInit(0)` 成功、设备数为 1，现有 CUDA 12.9 与驱动匹配。原生编译和 kernel/复制/CUPTI 专项通过，补充内存插桩的未完成项也已单独记录。此结果提供可用的远程 GPU 验证环境，不代表 Linux 驱动已恢复或完整 NLP CUDA 能力已实现。
