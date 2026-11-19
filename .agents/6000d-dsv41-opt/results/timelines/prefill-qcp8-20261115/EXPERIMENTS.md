# QCP8＋EP8稳态Prefill短时Nsight Systems采集

沿用开启自动调优的基线。8卡、C192、输入128928、缓存121088、未命中7840、输出1、QCP8/EP8/KVP1、全组chunk16384。原生P计算与本地单token完成适配保持不变，不含KV传输。

完成128个预热请求后，通过原生CUDA_PROFILER控制截取12个forward。仅追踪CUDA和NVTX，关闭CPU采样和上下文切换。为encoder、CED narrowing、decoder、layer、attention、compressor和Engram增加轻量NVTX标签。标签只读取CPU可见的形状，不改变模型计算、不加CUDA同步。

采集结束后继续运行以确认稳定请求负载，再清空请求和释放8卡。Nsight父进程正常退出后保存原始报告。分析排除两端批次，保留原始nsys-rep不裁剪。采集期间的时长包含工具开销，不用于替换原始吞吐成绩。

状态：采集、导出、下载校验和分析均已完成。8卡已释放。没有修改REPORTS.md。

## 实测结论

这段QCP8＋EP8记录中，主要GPU耗时位于encoder。MoE融合内核是最大的内核时间类别。CED收缩后的本地查询行分布明显不均，但不能据此将整组吞吐差距归因于CED。

每张卡记录12个Prefill forward，共299,155个CUDA kernel事件。GPU事件跨度为5.416044秒。分析使用每卡中间8批，保留了全部原始报告。
11批的encoder每卡2048行，1批每卡2036行。每批均为3个extend请求、0个generation Decode请求。这是既有分批策略的正常小幅变化。

八卡平均批次周期为464.171ms。各卡周期中位数的平均值为436.828ms。均值受一次批次切换空窗影响，不能当作所有批次的典型周期。

## GPU时间构成

下表对每卡8批分别计算内核时间并集，再对8张卡取平均。类别可在不同stream上重叠，因此不能直接相加作为独占关键路径。

| 类别 | ms/卡/批 |
| --- | ---: |
| MoE融合计算与通信 | 201.161 |
| Attention | 71.603 |
| 显式集合通信 | 46.691 |
| Dense GEMM | 41.153 |
| 其他内核 | 24.644 |
| Gather / Embedding | 19.026 |
| Indexer / Top-k | 11.413 |

encoder相关内核时间并集为367.506ms/卡/批，约占内核活动时间的89.1%。CED narrowing为3.348ms，CED decoder尾部为39.743ms。这里的decoder是Prefill内部的CED阶段，不是生成阶段的Decode。

GPU0每批有20次`cake_megamoe_m128_nocopier`、20次`cake_megamoe_m64_nocopier`、47次NCCL AllGather和1次NCCL ReduceScatter。MegaMoE融合内核同时包含专家计算和通信等待，不能将其全部时长标为纯GEMM。显式集合通信时间也不能直接视为相对ADP的额外开销。

## CED后的行分布

| 中间批次 | GPU0至GPU7的本地查询行数 | 非零卡数 |
| --- | --- | ---: |
| 1 | 0、128、0、0、0、128、0、1 | 3 |
| 2 | 32、96、0、0、128、0、0、1 | 4 |
| 3 | 128、0、0、0、128、0、0、1 | 3 |
| 4 | 128、0、0、0、128、0、0、1 | 3 |
| 5 | 128、0、0、0、128、0、0、1 | 3 |
| 6 | 128、0、0、0、128、0、0、128 | 3 |
| 7 | 0、0、0、128、0、0、0、129 | 2 |
| 8 | 0、0、0、128、0、0、0、129 | 2 |

每批只有2至4张卡持有非零本地查询行。其余卡仍参与EP专家执行和集合通信，不等于这些卡完全空闲。NVTX的CPU区间时长包含异步提交和等待；GPU阶段耗时通过CUDA launch correlation计算，不能直接使用CPU NVTX时长。

## 批次切换空窗

第5至第6个选中批次之间，GPU0的forward内核边界间隔为188.558ms。逐卡对齐，并加入forward范围之外的准备内核、输出内核、memcpy和memset后，8卡共同没有已记录GPU活动的最长区间为 **159.153335ms**。

该区间位于原始报告时间轴3039.450963至3198.604298ms。下一批encoder为每卡2036行。这里只记录相关性，未证明行数变化导致空窗。正式观察窗口没有JIT编译、Decode或混合批次。当前记录没有CPU调用栈和上下文切换，不能区分调度、其他主机工作或Profiler扰动。

原始记录和8批总览保留了该空窗。没有删除异常批次来提高统计值，也未将这次短时采集的吞吐替换为无Profiler成绩。

## 可复核文件

- [原始Nsight Systems报告](hot/profile/trace.nsys-rep)，采集版本2026.5.2。使用该版本打开，在CUDA GPU视图查看GPU0至GPU7；在NVTX视图搜索`forward_step ext=3 dec=0`和`qcp.encoder`、`qcp.decoder`。
- [8批总览](analysis/steady-prefill-overview.png)；[单批放大](analysis/steady-prefill-timeline.png)。总览以2ms为一格，显示该格内内核时间最多的类别；原始报告保留准确事件和stream。
- [机器可读分析](analysis/timeline-analysis.json)、[采集核查](analysis/CAPTURE_AUDIT.json)、[逐卡摘要](analysis/rank-summary.csv)。
- Nsight导出的NVTX、CUDA kernel和CUDA API统计位于`hot/profile/summary_*.csv`。这些整段统计包含边界批次，与中间8批统计的范围不同。

全部423个请求验证为输入128928、缓存121088、输出1。采集前已完成128个预热请求。8卡均完成原生自动调优。下载文件与远端SHA256一致；清理后8卡显存均为0MiB，2597个冻结源文件未变，原有Gateway进程保留。

## 结论范围

本次只采QCP。它给出了QCP内部的时间构成、CED行分布和一处批次切换空窗。要量化QCP相对ADP的新增成本，需要同条件ADP时间线对照。当前证据不能把18.7%的吞吐差距全部归因于显式通信、CED或这一次空窗。
