# DeepSeek V4.1 indexer scores

## SM120 selection

The optional FlashInfer solution serves MXFP4 index caches on SM120 with 32
total query heads. It scores independent history tiles, then uses native
graph-safe TopK. It retains the portable path's BF16 dot rounding, weighted
head rounding, per-TP-shard rounding, and rank-order FP32 shard sum.
Boundary ties prefer the smaller logical ID. Reindex sorts candidate blocks
before scoring, so this rule also holds for unsorted caller inputs. Selected
IDs are sorted and padded with -1; invalid pages never become candidates.

The score and block-score matrices share a 128 MiB budget per query tile.
Their width follows table capacity for Full and candidate count for Reindex.
Tables wider than the native selector's 2**21-column limit, and score tiles
smaller than 16 rows, retain the portable implementation. Cache values,
score matrices, and selections remain invocation-owned. Graph replay reads
updated page tables, visible lengths, candidates, and query values.

Full scoring shares a 128-row history tile between two queries when their
page IDs match. Broadcast page tables use one shared scoring launch.
Distinct page-table rows use shared scoring for query tiles with at least
eight rows; smaller tiles retain independent scoring. The first launch
checks the current physical IDs and scores matching pages. A second launch
scores the remaining pages. Both launches read updated IDs on graph replay.
Each query still applies its own visible length. This
geometry preserves the existing FP32 head-reduction order and BF16 rounding
boundaries. Candidate reindexing, single queries, and smaller requested score
tiles keep independent scoring.
The shared path uses the same score budget and invocation-owned outputs.
SM120 scoring consumes the existing MXFP4 query and key codes directly with
native FP4 matrix instructions and FP32 accumulation. Query packing uses the
same codec as the portable path. It omits the intermediate query unpack and
per-tile key conversion. The dot result still rounds to BF16 before head
weighting. Scoring reduces each contiguous head shard once. It retains each
shard's rounding boundary and adds the partial scores in rank order.

## Datacenter native selection

The native Blackwell DeepGEMM indexer writes BF16 logits and passes BF16 rows
through masking, alignment, candidate gathering, and DeepSelect. Head weights
retain their existing precision. Paged scoring uses the caller's query tile;
replicated-cache decode supplies the complete local query batch, including
speculative verify rows. Owner-local history retains its per-request windows.
This design avoids splitting a decode batch solely because its configured
history capacity is large. Each score tensor uses 2 bytes per query-by-history
entry; native logits and their cleaned copy can coexist, so peak storage
includes the score matrices, alignment, and selection scratch.
Dense broadcast prefill retains its 128 MiB score budget and the runtime's
prefill query windows. The sparse candidate scorer already produces BF16 logits.
Kernel warmup uses the same output dtype as execution and compiles both
independent and request-grouped paged scoring before capture.

Invocation-owned index metadata can also carry per-query request IDs. Paged
Full passes the same IDs to schedule construction and scoring, allowing the
native kernel to reuse history across consecutive queries of one request.
Such rows share a page table and have nondecreasing visible lengths. Empty
rows get distinct scheduling labels so they cannot truncate an active request's
history bound. The runtime groups canonical decode rows and ordered prefill
spans. Free decode windows use independent rows so reordering queries cannot
truncate the history scored for an earlier row.
Request IDs, tables, and lengths remain unchanged while sharing metadata;
graph replay refreshes their captured producers.

The paged selection cleaner writes its final aligned score layout directly.
It masks invalid pages and initializes the last live tile's prefetch padding;
wholly future tiles are unspecified. Native token selection consumes this
buffer with exclusive row ends, avoiding another alignment copy. Candidate
block reduction similarly initializes only live blocks and their final aligned
segment, and native selection uses the valid block counts. The full-width
cleaner and generic score-preparation helpers retain their existing contracts.

Candidate block maxima remain FP32, with the existing newest-block pinning and
logical-position mapping. Converting BF16 scores to FP32 for this reduction
does not recover precision lost when the producer wrote BF16. Equal BF16
scores may select different valid boundary ties from an FP32-score Top-K.
Correctness checks therefore compare selected score multisets against the
same BF16 input, while model-quality checks must assess the precision change
separately. The Hopper score producer retains FP32 output.

## SM120 selected attention

The model obtains its decode query geometry from
`selected_attention_query_heads`. SM120 uses the real heads on each rank.
An eight-head TP shard therefore avoids computing 56 padding heads. The
existing in-place RoPE path rotates those real heads, and the sink keeps its
real head count. Other targets keep the existing 64/128-head geometry.
The explicit `rope_pad_query` operation still returns 64 or 128 heads.

The SM120 Triton implementation reads SWA and global pages directly. It
rounds decoded KV to BF16, matching the portable gather boundary. A single
query with up to 16 heads uses FP32 vector reductions. Other shapes share KV
reads across groups of 16 heads. Their QK uses BF16 tensor-core products with
FP32 accumulation. PV represents each FP32 probability as a BF16 high part
and a BF16 residual. Two BF16 tensor-core products accumulate into FP32.
The denominator still uses the original FP32 probabilities. This changes
the PV rounding and does not promise bitwise agreement with FP32 PV.
Native FP4 conversion decodes both nibbles before the BF16 KV boundary.
Grouped queries use a 16-row history tile. Small batches with one head group
use at least 16 history splits.

The selected history is split across CTAs. Each partial carries its FP32
maximum, denominator, and output numerator. One partial includes the sink.
The merge rescales these partials into a joint softmax. Empty selections
produce zero output and, when requested without a sink, negative-infinity
LSE. Duplicate slots retain their multiplicity.

Scratch is owned by each invocation and bounded to 128 MiB per query tile.
Page-table widths, visible lengths, and physical slots remain runtime data.
The split count uses four occupancy buckets. Compact prefill workspaces and
V4 cache layouts keep the portable implementation. Cache ownership and writes
remain unchanged.
