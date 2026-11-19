# 优化结果（最终，2026-11-18）

目标见 `OPTIMIZATION_GOAL.md`，逐内核 SOL 见 `SOL_ANALYSIS.md`。本文是金标准实测的最终成绩、每项改动的状态和核对结论。
全部运行已结束，节点上没有留下进程。

## 结论

模型 DeepSeek-V4.1-Flash，两台 8× RTX 6000D。成绩分三档报，区别只在数值上改了多少：

- **逐位相同**：输出每一位都不变。
- **同精度**：运算顺序或中间舍入变了，误差量级不变，但不是逐位相同。
- **改精度**：误差变大（`wo_a` 和 draft LM head 改走 MXFP8）。

| 金标准 | 基线 | 逐位相同 | 同精度 | 改精度 |
| --- | ---: | ---: | ---: | ---: |
| P-only ADP8（req/s） | 5.688 | 7.341（+29%） | 8.991（+58%） | 9.570（+68%） |
| P-only QCP8（req/s） | 4.596 | — | 7.168（+56%） | — |
| D-only D16，交接配置（每卡 tok/s） | 1108.6 | 1203.0（+8.5%） | 1425.2（+28.6%） | 1488.1（+34.2%） |
| D-only D16，放开 KV 池（每卡 tok/s） | — | 同上 | 1451.9（+31.0%） | 1540.2（+38.9%） |

- D-only 的每个数都满足平均 TPOT 低于 10 ms。“放开 KV 池”指引擎参数 `--max-total-tokens` 从 262144 改成 393216，
  让每张卡能同时放 14 路以上；其余配置不变。
- 表里各档的数字都由独立的核对者从原始记录重算过，没有发现无效的窗口、报错或回退。
- 金标准强制接受长度、不保存输出，所以它只证明速度。各档的数值结论来自内核级验证。
- 改动全部是默认关闭的开关，代码在两台节点的 `opt-20261117/trees/all7`。

**11 月 19 日补充**：

- 新的实验代码树 `opt-20261117/trees/all8` = `all7` 加一个更低精度的选项：Prefill 注意力的 PV 只做一遍 BF16
  （`TOKENSPEED_SM120_PREFILL_ATTN_PV=bf16` 配 `TOKENSPEED_SM120_PREFILL_ATTN_PIPELINE=1`，默认关）。
  注意力输出相对 FP64 的 L2 误差从 1.7e-3 变成 2.0e-3，归入“改精度”。内核测试 74 条全过。
- 金标准 P-only ADP8 用它（开关集 `p8-prec1.sh`）是 **9.755 req/s（+71.5%）**，两轮 9.759 和 9.752；
  同一晚在 `all8` 上不开它（`p7-prec.sh`）是 9.574，与上表的 9.570 一致，所以这一项的收益是 +1.9%
  （`all8` 第一次启动那一轮是 9.439，偏低，没有采用）。
  这几个数没有派独立核对者，只跑了金标准自带的分析。上表仍是 11 月 18 日的成绩。
- Decode 的代码路径在 `all8` 里没有变化。
- **Prefill 的调度重叠试过了**（下面“需要你决定的”第 3 项）：金标准 ADP8 上去掉 `--disable-overlap-schedule`，
  改精度档加 PV 单遍是 10.236 和 10.462 req/s（平均 10.349，比关着的 9.755 高 6.1%，相对基线 +82%），同精度档是 9.163（比 8.991 高 1.9%，只跑了一次）。
  这改的是交接文档的固定配置，没有并入上表；开了之后两轮之间相差 2.2%。开关集 `p8-prec1-overlap.sh`、`p7-same-overlap.sh`。
- 本地检出 `tokenspeed-vs-main` 的工作区已更新到 `all8`（比 `all7` 多改 4 个文件，仍在未暂存的优化层）。

最主要的收益来源：

- Prefill：专家按层重排加热专家副本（让 8 张卡的 MoE 负载变均匀，避开这张卡按 SM 计的算力配额），
  注意力内核重写，共享专家并入 MoE 内核。
- Decode：注意力换成 FlashInfer 稀疏 MLA，共享专家并入 MoE 内核，稠密 GEMM 换分块；
  每步变快之后，同样的 10 ms 里每张卡能多带几路。

## 代码在哪里，怎么开

- 最终代码：两台节点的 `/raid/local/tokenspeed-pd-1p1d/opt-20261117/trees/all7`，两台内容一致
  （Python 源码校验和都是 `9d43595e35d42be01b779b27ca292e9b`）。
- 相对实测代码（`trees/base`）：改了 22 个文件，新增 10 个文件（6 个源码、4 个测试），新增约 6100 行、删除约 120 行。
- 所有优化都是环境变量开关，默认全关。开关全关时 forward 路径与实测代码相同，只在启动预热时多一次很小的校验
  （各 rank 核对专家放置方案和几个开关是否一致）。
- 每档用的开关组合是一个小脚本，在 `opt-20261117/work/opt-env/`：

  | 用途 | 逐位相同档 | 同精度档 | 改精度档 |
  | --- | --- | --- | --- |
  | P-only ADP8 | `p7-bit.sh` | `p7-same.sh` | `p7-prec.sh` |
  | P-only QCP8 | — | `q7-same.sh` | — |
  | D-only D16 | `d7-bit.sh` | `d7-same.sh` | `d7-prec.sh` |

- 运行记录在 `opt-20261117/runs/<运行名>/`。11 月 18 日 16:10 之后启动的运行，目录里另有
  `engine-N-switches.txt`，记录引擎启动时读到的开关文件内容和哈希。
- 冻结目录、`-megamoe` 实测目录、venv、交接包都没有动。
- **本地检出 `tokenspeed-vs-main` 已叠上最终代码**（11 月 18 日），并且两层改动分开显示：
  - HEAD 仍是上游的 `ee5b468f1`，没有提交。
  - **暂存区 = 实测代码**（上游 → 实测代码这一层：22 个修改、17 个新增文件）。在 VS Code 的源代码管理面板里是“暂存的更改”。
  - **工作区 = 最终代码**。没有暂存的部分就是优化这一层：22 个修改的文件加 10 个新增（未跟踪）的文件，
    在面板里是“更改”；编辑器行号旁标出的只有优化改的行。
  - 已核对：暂存区与节点的 `trees/base` 逐文件一致（1352 个文件），工作区与节点的 `trees/all7` 逐文件一致
    （1362 个文件）；差别只有本来就没拷的 24 个预编译 `.so`。
  - 节点上的 `review-vs-main-20261116/`（之前采 timeline 用的那份检出）没有动，仍是实测代码。

  优化层改的 22 个文件：

  ```
  python/tokenspeed/runtime/engine/scheduler_utils.py
  python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py
  python/tokenspeed/runtime/models/deepseek_v41.py
  python/tokenspeed/runtime/models/deepseek_v41_dspark.py
  python/tokenspeed/runtime/models/deepseek_v41_engram.py
  python/tokenspeed/runtime/models/deepseek_v41_moe.py
  tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv4/README.md
  tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv4/__init__.py
  tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv4/triton.py
  tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/README.md
  tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/_triton/index_score.py
  tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/flashinfer.py
  tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/__init__.py
  tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/_deep_gemm/mega_moe_stage.py
  tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/_flashinfer/mega_moe.py
  tokenspeed-kernel/python/tokenspeed_kernel/ops/quantization/__init__.py
  tokenspeed-kernel/python/tokenspeed_kernel/ops/quantization/triton.py
  tokenspeed-kernel/python/tokenspeed_kernel/ops/residual/README.md
  tokenspeed-kernel/python/tokenspeed_kernel/ops/residual/__init__.py
  tokenspeed-kernel/python/tokenspeed_kernel/ops/residual/triton.py
  tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/flashinfer/mega_moe_sm120.py
  tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/flashinfer/mxfp8.py
  ```

  优化层新增的 10 个文件：

  ```
  python/tokenspeed/runtime/moe/ced_row_bound.py
  python/tokenspeed/runtime/moe/expert_file_placement.py
  tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv4/_triton/compact_variants.py
  tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/flashinfer/dsv41_mixed_decode.py
  tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/flashinfer/mxfp8_sm120_grouped.py
  tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/flashinfer/mxfp8_sm120_lm_head.py
  tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_flashinfer_sm120.py
  tokenspeed-kernel/test/nvidia/ops/test_attention_dsv41_index_grouped_sm120.py
  tokenspeed-kernel/test/nvidia/ops/test_attention_dsv4_prefill_pipeline_sm120.py
  tokenspeed-kernel/test/nvidia/ops/test_mhc_fused_sm120.py
  ```

## 改动清单

归档分三档：**逐位相同**（输出每一位都不变）、**同精度**（运算顺序或中间舍入变了，误差量级不变，但不是逐位相同）、
**改精度**（误差变大）。金标准强制接受长度、不记录输出内容，测不出输出对不对，所以每项的数值正确性都是在内核级验证的
（逐位比较，或对 FP64 参考比较）。

第一轮（按 SOL 差距从大到小做的）：

| 改动 | 开关（都以 `TOKENSPEED_` 开头） | 归档 | 用在 | 内核级验证 |
| --- | --- | --- | --- | --- |
| 按层重排专家所在的卡，最热的专家每张卡各放一份 | `MOE_EXPERT_PLACEMENT=<文件>` | 逐位相同 | P | 各种路由下输出逐位相同，含 CUDA Graph 回放 |
| mHC 两个内核合成一个 | `SM120_MHC_FUSED=1` | 逐位相同 | P | 逐位相同 |
| 跳过没人读的历史 gather | `SKIP_UNREAD_HISTORY_WORKSPACE=1` | 逐位相同 | P（只对 QCP8 有效） | 16 种启动组合逐位相同 |
| b12x GEMM 在 65–96 行改用 (32,128) 分块 | `SM120_MXFP8_TILE_32X128=1` | 逐位相同 | D | 117 种情况逐位相同 |
| Prefill 注意力的 PV 改成两项 BF16 拆分 | `SM120_PREFILL_ATTN_PV=bf16x2` | 同精度 | P | 0.19% 的值不同，最大偏差 2.0e-3；相对 FP64 的误差不变（1.66e-3） |
| 共享专家并入 MegaMoE 内核 | `MEGA_MOE_FUSED_SHARED=1` | 同精度 | P、D | 与分开计算相对差 0.29%；两者相对 FP64 都是 4.7%（MoE 自身的量化误差） |
| Decode 注意力换成 FlashInfer SM120 稀疏 MLA | `SM120_DSV41_ATTENTION_FLASHINFER=1` | 同精度 | D | 相对 FP64 误差 1.9–2.6e-3（原内核 1.6–1.7e-3） |
| `wo_a` 直接用 checkpoint 的 FP8 码走 MXFP8 | `SM120_WO_A_MXFP8=1` | 改精度 | P、D | 与现在相差 2.7%（均方根） |
| draft 那一遍的 LM head 走 MXFP8 | `SM120_LM_HEAD_MXFP8=draft` | 改精度 | D | logit 最大偏差 0.95；合成数据上 0.2–2.7% 的行 argmax 改变 |

第二轮（剩下的小项，全部逐位相同）：

| 改动 | 开关（都以 `TOKENSPEED_` 开头） | 用在 | 内核级实测 |
| --- | --- | --- | --- |
| Prefill 注意力内核改成流水式（要求先开 `PV=bf16x2`，输出与它逐位相同） | `SM120_PREFILL_ATTN_PIPELINE=1` | P | 2048 行 × 640 键：2455 → 1345 µs |
| Prefill 注意力输出不再先清零 | `PREFILL_ATTN_SKIP_ZERO_FILL=1` | P | 逐位相同 |
| 只 gather 被选中的历史 | `PREFILL_GATHER_SELECTED_HISTORY=1` | P | 逐位相同 |
| 索引打分按组共享历史 | `SM120_DSV41_INDEX_GROUPED=1` | P、D | 索引打分耗时：Prefill 每次 forward 合计 11.5 → 7.2 ms；Decode 每步 1.31 → 1.00 ms |
| CED 收窄之后的 MegaMoE 只按收窄后的行数启动 | `MEGA_MOE_DECODER_ROW_BOUND=1` | P（只 ADP8） | 每次 forward 约省 0.6 ms |
| FFN 的 mHC 系数挪到 router 之后提交 | `SM120_FFN_MIXES_AFTER_ROUTER=1` | D | 每步约省 0.11 ms |
| RMSNorm 量化直接写出 swizzle 后的 scale | `SM120_RMSNORM_SWIZZLED_SCALES=1` | D | 每步约省 0.25 ms |
| Engram 查表放到上一层 MegaMoE 的空档里做 | `ENGRAM_LOOKUP_UNDER_MEGAMOE=1` | D | 每步约省 0.17 ms |
| 每步的 block table 上传走单独的 stream | `BLOCK_TABLE_UPLOAD_STREAM=1` | D | 每步约省 0.055 ms |

“同精度”档算等价版还是改精度版，由你定。

## 没有采用的候选

| 候选 | 结论 | 依据 |
| --- | --- | --- |
| MegaMoE 同机传输改 LSA | 不采用 | 第一轮 QCP8 金标准 4.593 对 4.596 req/s。第二轮查到卡顿原因（NCCL 的包走了有损的流量类），改到无损类之后内核级每次 forward 快 3.4 ms，但 ADP8 金标准 8.642 对 8.615 req/s，在波动以内 |
| Engram 取数预取 | 保持关闭 | 第一轮在引擎里没有生效（辅助线程报错后退避）；修好之后确实生效，但金标准 8.648 req/s，没有可测的收益 |
| DeepGEMM 替换 b12x | 不可用 | FP8 路径在 SM120 上跑不起来 |
| Prefill 注意力换 FlashInfer | 不采用 | 每层 3212 µs，自研的两项 BF16 拆分是 2445 µs |
| Prefill 注意力每程序 32 头 | 不采用 | 逐位相同但没有变快（4364 对 4204 µs） |
| 4096 行一次 MegaMoE | 不采用 | 内核级实测慢 1%，已有的 ADP8 chunk 4096 运行慢 6.5% |
| NCCL 通道数等设置 | 没跑金标准 | 预期收益约 1 ms，在波动以内，且只对 QCP8 有效 |
| 索引打分的另一种分组写法（第二轮 c 项） | 没合入 | 与“按组共享历史”改的是同一处，后者更快 |
| Decode MegaMoE 的协议常数 | 没做 | 这是 Decode 每步剩余差距里最大的一块。要改 FlashInfer 生成的内核源码，生成器不在我们手里，而且加载时校验文件哈希 |

## P-only 详细

口径：节点 28 的 8 张卡，C192，每个请求 128928 个输入 token、其中 121088 个命中缓存（名义 94%），输出 1 个 token，
三个连续的稳态窗口取平均。树 `all7`。

ADP8（基线 5.688 req/s，运行 `p-adp-base-1117a`，开关全关）：

| 档位 | 各次运行（req/s） | 取值 | 相对基线 | 运行名 |
| --- | --- | ---: | ---: | --- |
| 逐位相同 | 7.341 | 7.341 | +29.1% | `p7-adp-bit-1118g` |
| 同精度 | 8.871、9.029、9.072 | 8.991（三次平均） | +58.1% | `p7-adp-same-1118g`、`p7-adp-same-b-1118g`、`p7-adp-same-c-1118h` |
| 改精度 | 9.565、9.574 | 9.570（两次平均） | +68.2% | `p7-adp-prec-1118g`、`p7-adp-prec-b-1118h` |
| 对照：同精度档去掉第二轮小项 | 8.599（另一次 8.326 不采用，见下） | 8.599 | +51.2% | `p7-adp-same-r1-1118g`、`p7-adp-same-r1-b-1118h` |

QCP8（基线 4.596 req/s）：

| 档位 | req/s | 相对基线 | 运行名 |
| --- | ---: | ---: | --- |
| 同精度 | 7.168 | +56.0% | `p7-qcp-same-1118g` |

说明：

- 每次运行的三个窗口都有效，窗口之间相差不到 2%；日志没有报错，没有任何优化回退。
- **重复性**：条件相同的重复（内核缓存是热的、调度形态正常）相差约 ±0.5%，例如改精度档 9.565 和 9.574。
  有两种情况会让单次运行偏低，都查到了：
  - 一棵树第一次启动引擎时偏低约 2%（同精度档的 8.871 就是 `all7` 的第一次启动；原因没有查到）。
  - 对照的第二次运行 8.326 偏低约 3%。原因在日志里看得到：8 个副本“开始下一个请求”的时刻，正常是落在每个请求周期的 2 个 forward 步里，
    这次启动时错开成了 3 个步，多出来的那一步拖慢了所有副本。这是启动时的调度结果，不是开关的性质；
    本地 14 次 ADP8 运行里只有这一次。所以这次不用来算收益。
- **第二轮小项的收益**：同精度档三次（8.871、9.029、9.072）对对照 8.599，约 **+4.5%**（+3.2% 到 +5.5%）。
- 各项的贡献（第一轮按顺序叠加时逐步实测的 req/s）：专家重排 5.688 → 6.511（+14.5%），注意力两项 BF16 拆分 → 7.461（再 +14.6%），
  共享专家并入 MoE → 7.833（再 +5.0%），热专家副本 → 8.624（再 +10.1%），`wo_a` 走 MXFP8 → 9.275（再 +7.5%）。
- 只跑了一次的两档（ADP8 逐位相同、QCP8 同精度）按 ±0.5% 到 ±2% 看。
- 专家放置方案是用这份负载自己的路由统计求解的，所以这是“针对这份负载调好之后”的成绩。
  换一份路由分布不同的负载，需要重新统计和求解（工具和开关都在树里）。
- 优化后的 timeline（第一轮结束时在树 `all5` 的同精度档上采的，数据在 `nsys-timelines/final-20261118/`）：
  ADP8 每卡每次 forward 从 365.4 ms 降到 237.4 ms。其中 MoE 融合内核 202.8 → 107.0 ms，注意力 71.6 → 39.5 ms。
  第二轮之后没有重新采 timeline。

## D-only 详细

口径：两台节点 16 张卡（DEP16），DSpark T=5，强制接受长度 5.35，每个请求输入 128928、输出 2048 个 token，
在所有请求共同的 decode 窗口里测。指标是“平均 TPOT 低于 10 ms 前提下，每卡输出吞吐的最大值”。树 `all7`。
基线是交接文档的 C176：每卡 1108.6 tok/s，9.901 ms（这次重跑：1104.5 tok/s，9.938 ms）。

### 各档的全部测点

“KV 池”一列是引擎参数 `--max-total-tokens`。交接配置是 262144，每张卡最多同时放 14 路，所以最高只能到 C224。
按你的决定（只要 TPOT 在 10 ms 以内就可以跑），另外用 393216 测了更高的并发；这个参数只改 KV 块池的大小，
每张卡多占约 123 MB 显存，不改任何计算。

| 档位 | 并发 | KV 池 | 平均 TPOT | 每卡输出吞吐 | 达标 | 运行名 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 逐位相同 | C176 | 262144 | 9.669 ms | 1134.9 tok/s | 是 | `d7-bit-1118g` |
| 逐位相同 | C180 | 262144 | 9.843 ms | 1140.5 tok/s | 是 | 同上 |
| 逐位相同 | C184 | 262144 | 9.863 ms | 1163.4 tok/s | 是 | `d7-bit-hi-1118h` |
| 逐位相同 | **C192** | 262144 | **9.951 ms** | **1203.0 tok/s** | 是 | 同上 |
| 同精度 | C176 | 262144 | 8.523 ms | 1288.4 tok/s | 是 | `d7-same-1118g` |
| 同精度 | C224 | 262144 | 9.816 ms | 1425.2 tok/s | 是（3 次里 1 次 10.026 ms） | 同上 |
| 同精度 | **C232** | 393216 | **9.973 ms** | **1451.9 tok/s** | 是，贴着上限（3 次里 1 次 10.011 ms） | `d7-same-cap-1118g` |
| 改精度 | C176 | 262144 | 8.204 ms | 1338.0 tok/s | 是 | `d7-prec-1118g` |
| 改精度 | C224 | 262144 | 9.393 ms | 1488.1 tok/s | 是 | 同上 |
| 改精度 | C232 | 393216 | 9.641 ms | 1500.5 tok/s | 是 | `d7-prec-cap-1118g` |
| 改精度 | C240 | 393216 | 9.715 ms | 1540.2 tok/s | 是 | 同上 |
| 改精度 | C248 | 393216 | 约 9.94 ms | 约 1557 tok/s | 不算数（见下） | `d7-prec-cap-1118g`、`d7-prec-cap2-1118h` |
| 改精度 | C256 | 393216 | 10.066 ms | 1585.9 tok/s | 否 | `d7-prec-cap-1118g` |
| 对照：同精度档去掉第二轮小项 | C224 | 262144 | 9.868 ms | 1415.9 tok/s | 是 | `d7-same-r1-1118g` |

### 每档的最好成绩

| 档位 | 交接配置（KV 池 262144） | 相对基线 | 放开 KV 池（393216） | 相对基线 |
| --- | --- | ---: | --- | ---: |
| 逐位相同 | C192：1203.0 tok/s，9.951 ms | +8.5% | 不需要（没有碰到 KV 上限） | — |
| 同精度 | C224：1425.2 tok/s，9.816 ms | +28.6% | C232：1451.9 tok/s，9.973 ms | +31.0% |
| 改精度 | C224：1488.1 tok/s，9.393 ms | +34.2% | C240：1540.2 tok/s，9.715 ms | +38.9% |

固定在基线的并发 C176 上比（看 TPOT 降了多少）：

| 档位 | 平均 TPOT | 每卡输出吞吐 | 相对基线 |
| --- | ---: | ---: | ---: |
| 基线 | 9.901 ms | 1108.6 tok/s | — |
| 逐位相同 | 9.669 ms | 1134.9 tok/s | +2.4% |
| 同精度 | 8.523 ms | 1288.4 tok/s | +16.2% |
| 改精度 | 8.204 ms | 1338.0 tok/s | +20.7% |

### 说明

- **TPOT 是按“每张卡几路”一级一级跳的。** 16 张卡每一步是同步走的，所以只要有一部分卡多了一路，所有请求的 TPOT 一起上一个台阶；
  同一台阶内再加并发，TPOT 几乎不动。例如改精度档：每卡 14 路（C224）9.39 ms，开始出现 15 路（C232）9.64 ms，
  全部 15 路（C240）9.72 ms，开始出现 16 路（C248）9.94 ms，全部 16 路（C256）10.07 ms。
  所以最划算的并发是 16 的整数倍：逐位相同档 C192（每卡 12 路）、改精度档 C240（每卡 15 路）。
- **逐位相同档从 +2.9% 提到 +8.5%**，靠的是第二轮小项把每步缩短了约 1.4%，使每卡 12 路刚好落在 10 ms 以内
  （第一轮的代码在 C184 是 10.013 ms，不达标）。余量只有 0.05 ms，六次重复全部低于 10 ms（9.93–9.97 ms）。
- **同精度档 C232 是贴着上限过的**：平均 9.973 ms，余量 0.027 ms，比两次启动引擎之间的差别（约 0.04 ms）还小，而且只测了一次启动。
  六次重复里五次低于 10 ms，任取三次的平均都低于 10 ms。如果想留余量，就用 C224 的 1425.2 tok/s。
- **改精度档的 C248 实际能跑到，但按基准工具的规则不算数。** 测了两次（两次启动引擎），12 次重复全部低于 10 ms
  （9.90–9.99 ms，平均 9.94 ms，每卡约 1557 tok/s，相对基线 +40.4%）。但基准客户端自己的采样线程在这个并发下停顿得更多，
  每次运行 6 次重复里都只有 2 次通过自检，而工具要求每次运行至少 3 次。所以正式成绩取 C240；
  如果接受把两次运行合起来算（4 次通过自检，平均 9.924 ms、1559.3 tok/s），改精度档就是 +40.7%。这一点由你定。
- **放开 KV 池的收益不大**：同精度档 +1.9%（C224 那个数里有一次偶发慢步；去掉它之后 C224 约 1440 tok/s，C232 只高约 1%），
  改精度档 +3.5%。原因是每卡加到 15、16 路时 TPOT 也跟着上台阶，TPOT 的余量很快用完。
- **第二轮小项在 Decode 上的收益**：同精度档 C224 的每步时间中位数从 52.94 ms 降到 51.96 ms（−1.9%，两次运行的各六次重复互不重叠）；
  但吞吐指标只差 +0.7%，在波动以内。逐位相同档在相同并发上是 +1.4% 到 +1.5%。
- **Decode 的波动**：同一配置重新启动引擎，结果相差 0.3–0.5%。C224 附近单次重复偶尔会偏 2–3%
  （客户端采样停顿或引擎偶发的慢步），所以每个点都是 3 次以上重复的平均。
- 优化后的 timeline（第一轮结束时在树 `all5` 的同精度档 C224 上采的）：每卡每步 52.8 ms，与基线 C176 的 52.9 ms 相同，
  但每卡从 11 路变成了 14 路。Decode 注意力从每步 6.3 ms 降到 3.2 ms，稠密 GEMM 从 7.6 ms 降到 4.7 ms；
  MoE 融合内核因为行数变多从 29.8 ms 升到 34.6 ms，是剩下的最大一块。

## 独立核对

运行记录都拷到本地，由核对者从原始测量文件重算。核对者不接触节点，只读本地拷贝。

- 主要运行（Prefill 6 次、Decode 主链 4 次）：每次运行一位核对者重算，另一位专门挑错的复核者再查一遍前一位的结论。
- 后加的运行（Prefill 3 次重复、大 KV 池 2 次、逐位相同档 C192）：每次运行一位核对者重算；
  其中贴着上限的同精度档 C232 另派了复核者。
- 没有独立核对的两个点：逐位相同档的 C184（不是最好成绩）和改精度档 C248 的补测（不计入正式成绩）。
  这两个只用同一套脚本做了检查。

核对结论：

- **数字**：所有运行的每个窗口、每次重复都从原始文件重算，与汇总结果一致。
- **有效性**：计入成绩的点按基准自己的规则全部有效（改精度档 C248 不算数，见 D-only 一节）。没有报错，没有任何优化回退（流水式注意力内核不可用、放置方案被拒、
  FlashInfer 注意力回退、各 rank 开关不一致、MXFP8 转换失败这几类日志都是 0 条）。
- **引擎命令行**：与交接文档规定的设置逐项一致，Decode 的命令行与基线运行逐字节相同
  （大 KV 池的运行只多了 `--max-total-tokens 393216` 这一个参数，并已在 16 个 rank 的日志里确认生效）。
- **开关是否真的开了**：共享专家融合、FlashInfer 注意力、专家放置、`wo_a` 和 LM head 的 MXFP8 在日志里有明确的行，每个 rank 都有。
  其余开关代码不打印日志，只有间接证据（启动时编译的内核个数、CUDA Graph 占用的字节数随开关组合变化）。
  18 日 16:10 之后启动的运行，目录里记录了引擎实际读到的开关文件内容，核对时与预期逐字节相同。

必须和成绩一起说明的事：

1. **档位靠内核级验证，不靠金标准。** 金标准强制接受长度、不保存输出，测不出输出对不对。
   “逐位相同”“同精度”的结论来自内核级的逐位比较和对 FP64 的比较。
2. **改精度档只是速度数字。** `wo_a` 和 draft LM head 走 MXFP8 之后，真实接受率和输出质量会不会变，没有测。
3. **运行之间有波动。** 条件相同的重复，Prefill ADP8 和 Decode 都在 ±0.5% 左右；Prefill 个别运行会偏低 2–3%
   （一棵树的第一次启动、启动时调度错开，见 P-only 一节）。小于 1% 的差别不应当算作收益。
4. **Prefill ADP8 每次运行（含基线）第 1 个窗口里都有 Triton 的 JIT 日志。** 交接文档“窗口内不能有 JIT”这一条，
   字面上基线和所有实验都不满足。缓存热的时候它们是 1–2 ms 的加载，对成绩的影响在 0.3% 以内。QCP8 的窗口是干净的。
5. **Decode 每个并发点 6 次重复里有 1–4 次被基准客户端自己的规则剔除**（客户端的采样线程在窗口内停顿超过 200 ms）。
   停顿发生在客户端，引擎的节奏没有变；被剔除的重复和保留的重复数值几乎相同，剔除没有让成绩变好看。
   基线运行也是同样的情况。一个并发点至少要 3 次保留的重复才算数。
6. **Decode 的指标是“平均 TPOT 低于 10 ms”，不是“每次重复都低于 10 ms”。** 贴着上限的点会有个别重复高于 10 ms，
   下面的表里逐个标出。
7. **专家放置方案针对这份负载求解**，换负载要重新求解。
8. **基线**：Prefill 用这次在同一节点重跑的基线（ADP8 5.688、QCP8 4.596 req/s，QCP8 的历史成绩是 4.632）；
   Decode 用交接文档的历史成绩 1108.6 tok/s（这次重跑是 1104.5，低 0.4%）。基线都只有 C176 一个并发点。

## 代码审查和验证

金标准测不出输出对不对，所以代码本身另外查了三道：

1. **第一轮合并后的独立审查**（树 `all4`）：8 位审查者各看一块（专家放置和副本、共享专家融合、Decode 注意力、
   Prefill 注意力、稠密 GEMM、Engram 和 mHC、合并完整性、功能间相互影响）。
   - 会在 RTX 6000D 上算错或挂起的缺陷：没有发现。
   - 查出并已修的问题：Engram 预取的辅助线程没有选定 CUDA 设备；GEMM 分块开关会让内存里的自动调优缓存失效（只影响冷启动）；
     放置文件在非 SM120 的 MegaMoE 后端上应当拒绝；EP 组内没有校验各 rank 的开关是否一致；Engram 预取失败时没有退避。
     修复在 `all5`、`all6` 里完成，修完重新做了内核级验证，修复前后金标准成绩相差不到 1%。
2. **第二轮每个原型**：作者之外另有一位验证者独立复测（逐位比较和计时都重做），然后才合并。
3. **合并成 `all7` 之后**：全部开关关闭时，所有探针的输出摘要与 `all6` 相同；四个原型各自的基准在 `all7` 上重跑，
   结果与各自报告一致；另外补测了作者没测过的组合（Prefill 三个注意力开关加索引分组；真实 MegaMoE 内核 EP8 上的回调）。
   合并时发现一处单元测试回归（4 个用例，测试不经构造函数建模型时缺三个属性），已用三个类级默认值修掉，
   修后测试结果与 `all6` 相同。引擎不受这个问题影响。

仍然没有覆盖到的：

- FlashInfer Decode 注意力和索引分组在同一次 forward 里的组合，内核级没有单独测；它们只在引擎的金标准运行里一起跑过
  （无报错、无回退，但金标准不检查输出）。
- 第二轮 Decode 的四个小项，内核级收益是用替身 MegaMoE 测的，合计约 0.5 ms 每步；在引擎里只能看到合计效果。

## 没做的，和需要你决定的

没做的：

- **Decode MegaMoE 的协议常数。** 这是 Decode 每步里剩下的最大一块（每步约 52 ms 里 MoE 融合内核占约 35 ms，
  SOL 分析里它离理论值最远）。要改 FlashInfer 生成的内核源码，生成器不在我们手里，加载时还校验文件哈希。
- **第二轮之后的 timeline** 没有重新采（现有的优化后 timeline 是第一轮结束时的）。
- **真实接受率和输出质量**：金标准强制接受长度，改精度档对真实接受率的影响没有测。
- **端到端 PD**：按之前的决定，这两台节点上不测。

需要你决定的：

1. **“同精度”档算等价版还是改精度版。** 它的输出不是逐位相同，但相对 FP64 的误差量级没有变。
   如果算等价版，对外的“等价”成绩就是同精度档；如果不算，就是逐位相同档。
2. **Decode 对外报哪个 KV 池配置。** 交接配置（262144）下的成绩和放开到 393216 的成绩都在上面。
   同精度档放开后只多 1–2%，而且贴着 10 ms 上限；改精度档多 3.5% 以上。
3. **Prefill 的 overlap 调度。** 金标准的固定配置里是关的（`--disable-overlap-schedule`）。打开它可能还有收益，
   但改的是金标准规定的配置，所以没有动。
   **11 月 19 日已实测**：打开后改精度档 +6.1%、同精度档 +1.9%（见“结论”一节的补充）。要不要把它算进正式成绩，由你定。
