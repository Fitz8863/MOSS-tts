# MOSS-TTS K3 项目 AI 交接说明

> 本文件面向后续 AI/智能体对话窗口。开始工作前请先阅读本文件，再检查真实工作区、板端状态和 Git diff；不要仅依据本文件声称“已验证”。

## 1. 项目定位

本项目是在 SpaceMIT K3 开发板上部署 MOSS-TTS-Nano 的 **C++ + ONNX Runtime + ONNX 模型**路线。

当前目录：

```text
本地 SSHFS 工作区：/home/heweijie/spacemit-k3-dev/projects/MOSS-tts
板端真实目录：    /home/spacemit/projects/MOSS-tts
```

当前明确不使用：

- Python 推理入口
- GGUF
- llama.cpp / ggml
- 本目录中的 A100/IME2 ONNX 推理路线

相邻的 GGUF/llama.cpp 路线位于板端：

```text
/home/spacemit/projects/MOSS-tts-llamacpp
```

## 2. Git 与工作状态

当前分支：

```text
MOSS-tts-cpu
```

远端：

```text
target = git@github.com:Fitz8863/MOSS-tts.git
```

最近已提交的基础版本：

```text
b40aac3 restore C++ ONNX FP32 INT8 deployment
```

当前交接时工作区有未提交修改，主要是最近的“共享 ORT 线程池 / 8 核 8 线程”改动：

```text
M BOARD_DEPLOYMENT.md
M README_K3_ONNX.md
M cpp/src/main.cpp
M setup_k3_cpp_env.sh
```

接手后先运行：

```bash
cd /home/heweijie/spacemit-k3-dev/projects/MOSS-tts
git status --short --branch
git diff --check
git diff
```

除非用户明确要求，不要直接覆盖或丢弃这些未提交修改，也不要自动 push。

## 3. 板端连接

当前已知 SSH 主机：

```bash
ssh spacemit@spacemit-k3
```

解析到的板端主机名：

```text
bianbu-spacemitk3picoitx
```

板端用户：`spacemit`

用户曾提供 sudo 密码：`bianbu`。仅在确实需要 sudo 且用户授权的情况下使用，不要把密码写入脚本、日志或 Git。

每次远程操作先核对：

```bash
ssh spacemit@spacemit-k3 '
  hostname
  pwd
  cd ~/projects/MOSS-tts
  git status --short --branch
'
```

## 4. 当前推理架构

入口链路：

```text
run_k3_tts.sh
  -> setup_k3_cpp_env.sh
  -> taskset -c 0-7
  -> cpp/build-k3/moss-tts-onnx
  -> vendor ONNX Runtime CPUExecutionProvider
```

C++ 主程序：

```text
cpp/src/main.cpp
```

它加载 4 个 ONNX Session：

1. prefill
2. decode step
3. local fixed sampler
4. codec decode full

推理流程是自回归的，Session 在常驻模式下只初始化一次；后续输入文本只执行推理并覆盖同一个 WAV 文件。

当前输出：

```text
48 kHz
双声道
PCM16 WAV
```

当前 RTF 定义：

```text
RTF = 推理 wall time / 输出音频时长
RTF < 1 才表示整体快于实时
```

## 5. CPU、亲和性和线程的关键事实

板端 CPU 划分：

```text
CPU 0-7   = X100
CPU 8-15  = A100（用户称为 AI 加速算力核）
```

但当前普通用户 session 的系统级限制是：

```text
Cpus_allowed_list = 0-7
```

因此当前项目默认只能验证 X100 CPU 路线；不能把结果称为 A100/IME2 推理。即使使用 sudo，也不能简单突破当前 cpuset 限制。

当前默认环境：

```bash
MOSS_CPU_AFFINITY=0-7
MOSS_CPP_THREADS=8
MOSS_MAX_NEW_FRAMES=375
MOSS_VOICE=Junhao
MOSS_SEED=1234
```

### 线程实现的重要变化

旧实现给每个 Session 单独设置 `SetIntraOpNumThreads(N)`。因为有 4 个 Session，配置 `N=8` 时会创建多个独立线程池，进程 task 数约 29，并不是真正的进程级 8 线程。

当前未提交代码已改为：

- `Ort::ThreadingOptions`
- `CreateEnvWithGlobalThreadPools`
- `SetGlobalIntraOpNumThreads(threads)`
- `SetGlobalInterOpNumThreads(1)`
- 所有 Session 调用 `DisablePerSessionThreads()`

这样 4 个 Session 共用一个进程级 ORT 线程池：

```text
threads=4 -> 约 4 个进程 task
threads=8 -> 约 8 个进程 task
```

这才是当前语境下的“8 核 8 线程”。`inter-op` 仍然是 1，实际瞬时忙碌线程数仍取决于算子并行度。

## 6. 模型

默认模型目录：

```text
models/MOSS-TTS-Nano-100M-ONNX/
models/MOSS-Audio-Tokenizer-Nano-ONNX/
```

INT8 对照模型：

```text
models/MOSS-TTS-Nano-100M-INT8/
```

模型二进制文件通常被 `.gitignore` 忽略，由板端单独 provision。不要因为 Git 中看不到 ONNX 文件，就误判板端没有模型；应在板端检查：

```bash
find ~/projects/MOSS-tts/models -maxdepth 3 -type f | sort
```

模型选择：

```bash
./run_k3_tts.sh --model fp32 ...
./run_k3_tts.sh --model int8 ...
```

`setup_k3_cpp_env.sh` 中：

```text
fp32 -> MOSS-TTS-Nano-100M-ONNX
int8 -> MOSS-TTS-Nano-100M-INT8
```

当前 INT8 模型是社区动态 MatMul INT8 ONNX，不能先验保证一定比 FP32 快；必须在板端同条件测量。

## 7. 中文、英文和音色能力

当前 C++ CLI：

- 支持中文
- 支持英文
- 使用 SentencePiece C++ API
- 中文和英文走同一组 ONNX 图
- 支持 manifest 中内置命名音色，例如 `Junhao`、`Ava`（实际可用名字以 manifest 为准）

切换音色示例：

```bash
./run_k3_tts.sh --model int8 --voice Junhao '中文音色测试。' outputs/junhao.wav
./run_k3_tts.sh --model int8 --voice Ava 'English voice test.' outputs/ava.wav
```

`MOSS_VOICE=NAME` 仍然兼容，但命令行 `--voice NAME` 优先。当前 manifest 内置音色为：

```text
中文倾向：Junhao, Zhiming, Weiguo, Xiaoyu, Yuewen, Lingyu
英文倾向：Trump, Ava, Bella, Adam, Nathan
日文倾向：Soyo, Saki, Mortis, Umiri, Mei, Anon, Arisa
```

音色名称严格匹配，非法名称会报错；常驻进程启动后音色固定，不能按输入行动态切换。

当前 C++ CLI 尚未接入参考音频 encode，因此不要宣称已经支持任意参考音频克隆或任意音色迁移。当前承诺范围是 manifest 内置 prompt/audio codes 的命名音色。

## 8. 常用命令

### 编译

建议在板端编译，不能在本地 x86 主机直接用默认配置编译 vendor ORT 路线：

```bash
ssh spacemit@spacemit-k3
cd ~/projects/MOSS-tts
source ./setup_k3_cpp_env.sh
./build_k3_cpp.sh
```

构建依赖：

```text
/usr/include/onnxruntime_cxx_api.h
/usr/lib/python3.14/dist-packages/onnxruntime/capi/libonnxruntime.so.1.24.2+spacemit.a1
board_root_spm/usr/lib/riscv64-linux-gnu/libsentencepiece.so.0
```

构建后检查：

```bash
ldd cpp/build-k3/moss-tts-onnx | grep -E 'onnxruntime|sentencepiece|not found'
readelf -A cpp/build-k3/moss-tts-onnx | grep -E 'Tag_RISCV_arch|vector' || true
```

### 单次推理

```bash
cd ~/projects/MOSS-tts

./run_k3_tts.sh --model fp32 \
  '你好，这是 K3 上的中文 C++ ONNX 测试。' \
  outputs/zh_fp32.wav

./run_k3_tts.sh --model int8 \
  'Hello, this is an English C++ ONNX test on K3.' \
  outputs/en_int8.wav
```

### 常驻模式

```bash
cd ~/projects/MOSS-tts

./run_k3_tts.sh --model fp32 --interactive outputs/interactive_fp32.wav
./run_k3_tts.sh --model int8 --interactive outputs/interactive_int8.wav
```

然后输入文字并回车：

```text
你好，这是常驻模式测试。
Hello, this is a resident mode test.
exit
```

程序特征：

- 只打印一次 `initialized_once`
- 每次输入回车后生成音频
- 输出 WAV 覆盖上一次文件
- 每次打印 `frames/audio/wall/RTF`
- `exit`、`quit`、`:q` 或 Ctrl-D 退出

### 4 线程对照

```bash
MOSS_CPP_THREADS=4 \
  ./run_k3_tts.sh --model int8 --interactive outputs/interactive_int8_t4.wav
```

## 9. 已验证的共享线程池性能

测试位置：

```text
bianbu-spacemitk3picoitx:/home/spacemit/projects/MOSS-tts
```

测试条件：

```text
常驻模式
相同中文文本：欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。
MOSS_SEED=1234
MOSS_MAX_NEW_FRAMES=100
affinity=0-7
每个模型连续生成两次
测试期间暂停了另一个旧常驻进程以排除负载干扰
```

结果：

| 模型 | ORT 共享 intra-op | 第 1 次 RTF | 第 2 次 RTF | 平均 RTF |
|---|---:|---:|---:|---:|
| FP32 ONNX | 4 | 2.53742 | 2.52702 | 2.53222 |
| FP32 ONNX | 8 | 2.33129 | 2.24040 | 2.28585 |
| INT8 ONNX | 4 | 1.91039 | 1.89772 | 1.90406 |
| INT8 ONNX | 8 | 1.82399 | 1.76979 | 1.79689 |

相对 4 线程：

```text
FP32：8 线程平均快约 9.7%
INT8：8 线程平均快约 5.6%
```

结论：当前默认切换为 8 个共享 ORT intra-op 线程是合理的，但 X100 路线 RTF 仍大于 1，尚未实时。

## 10. 当前已知问题与证据边界

1. **没有 A100 推理证据**
   - 当前所有有效测试均为 CPUExecutionProvider + X100 CPU 0-7。
   - 不要将 `-march=rv64gcv` 或 `-mtune=spacemit-x100` 直接写成“已使用 A100/NPU”。

2. **RVV 只能说编译和运行环境具备条件**
   - C++ 主程序按 `-march=rv64gcv` 编译。
   - vendor ORT 是 SpaceMIT 版本。
   - 具体 ONNX 算子是否命中 RVV kernel，需要 vendor profile、kernel 文档或更强的运行证据。

3. **当前还没有 RTF<1**
   - 8 线程只是相对 4 线程更快，不能称为实时。

4. **旧常驻进程可能干扰测试**
   - 交接时板端曾存在旧进程 PID `1251266`：旧版 INT8、`--threads 4`、交互模式。
   - 当前它处于 `T`（stopped）状态，终端为 `pts/2`。
   - 后续测试前检查：
     ```bash
     pgrep -af 'moss-tts-onnx'
     ps -o pid,ppid,state,stat,tty,pcpu,cmd -p 1251266
     ```
   - 不要擅自 kill 用户交互进程；如需清理，先告知用户或确认其不再需要。

5. **本地与板端不是同一执行环境**
   - 本地工作区是 SSHFS 映射。
   - vendor ORT 路径通常只在板端存在。
   - 最终编译、运行和性能结论必须在板端验证。

## 11. 接手后的推荐流程

### 如果用户要继续优化性能

1. 检查 `git status`、板端 PID、`Cpus_allowed_list`。
2. 明确测试是单次模式还是常驻模式；用户偏向常驻模式。
3. 固定文本、seed、`MOSS_MAX_NEW_FRAMES`、模型和 affinity。
4. 一次只改变一个变量：线程数、模型、codec、ORT 配置等。
5. 记录：初始化时间是否包含、推理 wall、音频时长、RTF、task 数量、CPU mask。
6. 不把模型-only、decode-only、codec-only、端到端 RTF 混为一谈。
7. 使用 bounded timeout，并确认远端进程确实退出或恢复。

### 如果用户要提交

先做：

```bash
cd /home/heweijie/spacemit-k3-dev/projects/MOSS-tts
bash -n setup_k3_cpp_env.sh run_k3_tts.sh run_k3_tts_interactive.sh
git diff --check
git diff
```

然后在板端重新编译并做至少一次 smoke/interactive 验证，再由用户确认后提交和 push。提交时特别注意不要把以下内容加入 Git：

- ONNX 大模型
- `board_root_spm/`
- `cpp/build-k3/`
- `outputs/`
- 私有凭据

## 12. 重要文档

项目内已有：

```text
README.md
README_zh.md
README_K3_ONNX.md
BOARD_DEPLOYMENT.md
```

其中：

- `README_K3_ONNX.md`：当前 C++ ONNX 路线、命令、模型、性能和线程池说明。
- `BOARD_DEPLOYMENT.md`：板端部署与验证记录，按日期追加。
- `cpp/src/main.cpp`：真实推理实现和 ORT Session/线程池配置。
- `setup_k3_cpp_env.sh`：模型选择、线程、CPU affinity、动态库环境。
- `run_k3_tts.sh`：统一运行入口。
- `run_k3_tts_interactive.sh`：常驻模式兼容包装器。
