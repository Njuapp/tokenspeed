# Nsight Systems 采样

两份采样：Decode 是 2026-11-16 用 review 检出新采的，Prefill 是 11 月 15 日已有的 QCP8 采样。
ADP8 Prefill 没有采样。

`.nsys-rep` 用 Nsight Systems 2026.5.2 打开，在 CUDA GPU 视图看 GPU0–GPU7。
`.svg` 是从同一份数据画的概览，浏览器或 VS Code 的图片预览都能打开。

## 文件

| 路径 | 内容 |
| --- | --- |
| `decode-dep16-review-20261116/trace.nsys-rep` | Decode 原始报告，49 MB，SHA256 `ebfdbbba…12f525` |
| `decode-dep16-review-20261116/one-step-timeline.svg` | 一个 forward（52.9 ms）里 8 张卡的内核分布，外加 GPU0 上连续两层的放大 |
| `decode-dep16-review-20261116/kernels-by-origin.csv` | 每个 GPU kernel 的次数、时长、占比和来源 |
| `prefill-qcp8-20261115/trace.nsys-rep` | Prefill 原始报告，53 MB，SHA256 `3de2b771…4c81d0e` |
| `prefill-qcp8-20261115/one-batch-timeline.svg` | 一个 Prefill 批次（436 ms）里 8 张卡的内核分布 |
| `prefill-qcp8-20261115/steady-prefill-*.png`、`EXPERIMENTS.md` | 交接包里原有的 8 批总览、单批放大和采集记录 |
| `prefill-qcp8-20261115/kernels-by-origin.csv` | 同上，Prefill 的内核表 |

## Decode 的采集情况

- **代码**：`tokenspeed-vs-main` 检出（上游 `ee5b468f1` 加未提交改动），上传到两台节点的
  `/raid/local/tokenspeed-pd-1p1d/review-vs-main-20261116/`，引擎和客户端都从它导入
  `tokenspeed` 与 `tokenspeed_kernel`。FlashInfer 仍是 `656eaee4`，venv、权重、缓存目录不变。
  预编译的 24 个 `.so` 从实测目录拷入，因为检出里没有它们。
- **配置**：与 D16 基线相同。两台节点 16 卡，Attention DP16/TP1，MoE EP16，`all_gin`，DSpark T=5，
  强制接受长度 5.35，输入 128928、输出 2048，并发 176。
- **方法**：`.28` 的 8 个 rank 挂在 `nsys profile --trace=cuda,nvtx --cuda-graph-trace=node
  --capture-range=cudaProfilerApi` 下，关闭 CPU 采样和上下文切换。所有流都超过 600 个输出 token 后，
  客户端向 16 个 DP 副本各发一次引擎原生的 `CUDA_PROFILER` 请求，截取 40 个 forward。
  `.29` 的 8 个 rank 正常运行，不在报告里。
- **范围**：每张卡 40 个 forward、76,852 个内核事件，跨度约 2.12 秒；8 卡合计 614,817 个事件。
- **是否代表正常运行**：截取期间客户端测到的 TPOT 约 9.3–9.8 ms，触发前是 9.87 ms，forward 周期
  52.9 ms，与不挂 profiler 时一致。
- **不能当成绩用**：profiler 启动时全部流停顿约 1.3 秒，停止时停顿约 45 秒，所以这一轮的整窗
  TPOT（34.6 ms）无效。
- **对照**：同一流程先用节点上的 `-megamoe` 目录试跑了一次（运行目录 `decode-c176-nsys-1116a`），
  两份的内核事件数只差 1，各类占比相差不到 0.1 个百分点。
- **节点上的位置**：`handoff-claude-20261116/runs/decode-c176-nsys-review-1116a/profile/`
  （含 `trace.sqlite`、`command.txt`、`imported-from.txt`）。

## Prefill 的采集情况

沿用 11 月 15 日的采样，没有重采。详见同目录的 `EXPERIMENTS.md`，要点：

- **代码**：节点上的 `-megamoe` 目录。它的 Python 源码与 review 检出逐文件一致，只差 `version.py` 的版本号后缀。
- **配置**：`.28` 单机 8 卡，QCP8/EP8/KVP1，chunk 16384，并发 192，输入 128928，缓存 121088，输出 1，开启自动调优。
- **方法**：128 个预热请求之后截取 12 个 forward，只追踪 CUDA 和 NVTX。
- **范围**：8 卡合计 299,155 个内核事件，跨度 5.4 秒。

## 占比的口径

占比是“该 kernel 的时长之和 ÷ 全部 kernel 的时长之和”，8 张卡合计。不同 stream 上的 kernel 可以重叠，
所以各项相加略大于墙上时间。Decode 的 GPU 忙碌比例是 98.3%，Prefill 是 91%。
Prefill 的每批数字是 12 个批次的平均，包含首尾两个边界批次。

来源一列的含义：`megamoe` 是 MegaMoE 接入用到的内核，`sm120` 是节点本地 SM120 移植的内核，
`main` 是上游原有的 TokenSpeed 内核，`lib` 是 NCCL、cuBLAS、PyTorch 等外部库。

## 最终配置的采样（2026-11-18，`final-20261118/`）

两份都是在优化后的代码树 `opt-20261117/trees/all5` 上采的，方法与上面相同。原始报告和 SQLite 只在节点上
（本地配额不够），这里只有内核表和汇总：

| 文件 | 内容 |
| --- | --- |
| `adp8-final-kernels.csv`、`adp8-final-summary.json` | ADP8 Prefill，开关：8 个热专家副本的放置、注意力两项 BF16 拆分、共享专家并入 MoE |
| `adp8-baseline-kernels.csv` | 11-17 采的 ADP8 基线，用来对照 |
| `decode-final-kernels.csv`、`decode-final-summary.json` | D16 Decode，C224，开关：GEMM 分块、FlashInfer 注意力、共享专家并入 MoE |

节点上的位置（`.28`，`/raid/local/tokenspeed-pd-1p1d/opt-20261117/runs/`）：
`p5-adp-nsys-1118d/hot/profile/trace.nsys-rep`、`d5-nsys-1118d/profile/trace.nsys-rep`，旁边各有 `trace.sqlite`。
ADP8 基线的报告在 `p-adp-nsys-1117a/hot/profile/`。

要点：ADP8 一次 forward 从 365 ms 降到 237 ms（MegaMoE 202.8 → 107.0 ms，注意力 71.6 → 39.4 ms）；
Decode 在 C224 下一步是 52.8 ms，与基线 C176 时相同（注意力 6.15 → 3.11 ms，b12x GEMM 7.57 → 4.75 ms，
MegaMoE 因每卡行数从 66 增到 84 而从 29.8 增到 34.6 ms）。采样运行本身的吞吐不能当成绩用。

## b12x GEMM 按矩阵拆分（2026-11-18，`split_gemm.py`）

`split_gemm.py TRACE.sqlite|WINDOW.csv decode|prefill` 按每次 launch 在层里的位置（前后相邻的 Q 归一化、
注意力、路由 TopK、MegaMoE 内核）把 b12x MXFP8 GEMM 归到具体矩阵：`wq_a_wkv`、`wq_b`、索引器 `wq_b`、`wo_b`、
共享专家两次、Engram `wkv`、DSpark。只读。Decode 可以直接读节点上的 `trace.sqlite`（整段采样）；
Prefill 的层号靠 QCP 的 ReduceScatter 定位，所以只适用于 QCP 采样的单批 CSV。
