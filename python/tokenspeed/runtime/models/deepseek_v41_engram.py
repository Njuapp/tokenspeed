# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""DeepSeek V4.1 Engram, with caller-owned history and FP8-resident tables.

Integration contract:
* Construct one EngramHashState from the text config and the actual tokenizer.
  Its immutable input_parameters are borrowed by runtime preparation, which
  resolves raw previous-three IDs and computes hashes before model execution.
  -1 denotes a sequence boundary or a nonparticipating (e.g. image) token.
* Input preparation owns these small windows, including pending overlap tokens
  and the current speculative branch. Refresh pointer-stable input buffers for
  eager and graph forwards alike. Models receive prepared hashes and masks;
  never put request tensors in ForwardContext.
* Hidden states, hashes and masks share the query row layout. Without QCP they
  are replicated within mapping.attn.tp_group; QCP lookup gathers hash IDs and
  reduce-scatters the table contributions back to local query rows.
  GPU-resident tables are row-sharded across attention TP and all-reduced.
  Host tables stay in anonymous or shared host memory and gather through UVA.
  Their GPU registrations are released before their backing memory is freed.
  ``shared`` keeps one full copy per node and skips the lookup all-reduce.
  ``sharded`` keeps a host shard per attention-TP rank and retains the all-reduce.
  ``auto`` selects sharded when attention TP > 1. Output matches the GPU path.
  With TOKENSPEED_ENGRAM_HOST_PREFETCH=1 a pure-prefill forward over a shared
  host table gathers the rows on host threads and uploads them with one copy
  per table (EngramHostPrefetcher); every other case keeps the UVA gather.
* Intercept embed.weight/scale BEFORE a generic GPU weight iterator. Open the
  safetensors file on CPU, pass get_slice(name) to embed.load_sharded(), and mark
  the corresponding runtime parameter name loaded. GPU and host-sharded tables
  copy local rows; shared host tables copy every row on the writer rank.
  Do not cast the model wholesale to BF16: table codes/scales must stay bytes.
* wkv uses the caller-selected V4.1 Linear method for checkpoint scale
  expansion, hardware storage and post-load preparation. Engram owns the
  embedding loader and projection aliases, not a second projection loader.
  checkpoint_weight_aliases() maps raw checkpoint names, including
  wkv.scale, without renaming the embedding's per-row embed.scale.
"""

from __future__ import annotations

import atexit
import collections
import ctypes
import ctypes.util
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass

import numpy as np
import torch
from sympy import isprime
from tokenizers import Regex, normalizers
from tokenspeed_kernel.ops.embedding import host_gather
from tokenspeed_kernel.ops.metadata.ngram import (
    NGramHashParams,
    engram_hash_reciprocals,
)
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_ops import (
    all_reduce,
    prepare_all_reduce_lane,
    token_all_gather_rows,
    token_reduce_scatter,
)
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.fp8 import Mxfp8Config
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = get_colorful_logger(__name__)

_ENGRAM_EMBED_SUFFIXES = (".engram.embed.weight", ".engram.embed.scale")


def is_engram_embed_checkpoint_name(name: str) -> bool:
    """True for the two huge FP8 table tensors that must not be load_file'd."""
    return name.endswith(_ENGRAM_EMBED_SUFFIXES)


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Return raw-ID to normalized-ID lookup and its multiplier-defining size.

    ``tokenizer`` must expose the fast ``backend_tokenizer``; decoding deliberately
    includes special tokens and does not use HF whitespace cleanup.
    """
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_id: dict[str, int] = {}
    lookup = []
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        key = (
            backend.id_to_token(token_id)
            if "\ufffd" in text
            else normalizer.normalize_str(text) or text
        )
        lookup.append(key_to_id.setdefault(key, len(key_to_id)))
    return lookup, len(key_to_id)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    """Return CPU int64 [layers, lookbacks] odd multipliers, using reference RNG."""
    if not layer_ids or max_ngram_size < 2 or tokenizer_vocab_size < 1:
        raise ValueError(
            "Engram requires layers, at least two lookbacks and a vocabulary"
        )
    bound = max(1, (np.iinfo(np.int64).max // tokenizer_vocab_size) // 2)
    return torch.from_numpy(
        np.stack(
            [
                np.random.default_rng(10007 * layer_id).integers(
                    low=0, high=bound, size=(max_ngram_size,), dtype=np.int64
                )
                * 2
                + 1
                for layer_id in layer_ids
            ]
        )
    )


@dataclass(frozen=True)
class EngramLayout:
    """Prime bucket geometry in checkpoint layer/ngram/head order."""

    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    head_dim: int
    primes: tuple[tuple[tuple[int, ...], ...], ...]

    @classmethod
    def from_config(cls, config) -> EngramLayout:
        """Build and validate geometry from a V4.1 text config, not its wrapper."""
        layer_ids = tuple(config.engram_layer_ids)
        rows = tuple(config.engram_num_embeddings)
        if (
            not layer_ids
            or len(set(layer_ids)) != len(layer_ids)
            or len(rows) != len(layer_ids)
            or config.engram_max_ngram_size != 4
            or config.engram_n_heads < 1
            or config.engram_vocab_size < 2
            or config.engram_head_dim < 32
            or config.engram_head_dim % 32
        ):
            raise ValueError(
                "Invalid V4.1 Engram geometry (requires max ngram=4 and 32-wide blocks)"
            )
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(config.engram_max_ngram_size - 1):
                current, sizes = config.engram_vocab_size - 1, []
                for _ in range(config.engram_n_heads):
                    current += 1
                    while current in seen or not isprime(current):
                        current += 1
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        expected_rows = tuple(sum(sum(heads) for heads in layer) for layer in primes)
        if rows != expected_rows:
            raise ValueError(
                f"Engram table rows {rows} do not match prime buckets {expected_rows}"
            )
        return cls(
            layer_ids=layer_ids,
            num_embeddings=rows,
            max_ngram_size=config.engram_max_ngram_size,
            n_heads=config.engram_n_heads,
            head_dim=config.engram_head_dim,
            primes=tuple(primes),
        )


class EngramHashState(nn.Module):
    """Immutable tokenizer/hash constants only; no per-request token cache."""

    def __init__(self, config, tokenizer, device: torch.device | str):
        super().__init__()
        self.layout = EngramLayout.from_config(config)
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != config.engram_compressed_vocab_size:
            raise ValueError(
                f"Engram compressed vocabulary: got {vocab_size}, expected {config.engram_compressed_vocab_size}"
            )
        if not 0 <= config.engram_pad_token_id < len(token_map):
            raise ValueError("Engram pad token is outside the tokenizer vocabulary")
        self.pad_id = token_map[config.engram_pad_token_id]
        host_primes = torch.tensor(self.layout.primes, dtype=torch.int64, device="cpu")
        primes = host_primes.to(device=device)
        flat = primes.flatten(1)
        self.register_buffer("primes", primes, persistent=False)
        self.register_buffer(
            "reciprocals",
            engram_hash_reciprocals(host_primes).to(device=device),
            persistent=False,
        )
        self.register_buffer("offsets", flat.cumsum(-1) - flat, persistent=False)
        self.register_buffer(
            "token_map",
            torch.tensor(token_map, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "multipliers",
            compute_hash_multipliers(self.layout.layer_ids, 4, vocab_size).to(
                device=device
            ),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        previous_token_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Hash raw IDs [...] and lookbacks [..., 3] to [..., layers, hash columns].

        ``token_mask`` is bool [...] (False for padding/images). Past barriers and
        sequence starts are -1 in ``previous_token_ids``. Real IDs must be in the
        tokenizer vocabulary; callers must not pass clamped image placeholder IDs
        without their mask. All tensors live on the constants' device.
        """
        if (
            previous_token_ids.shape != (*input_ids.shape, 3)
            or token_mask.shape != input_ids.shape
        ):
            raise ValueError(
                "Engram IDs, previous-three window and mask shapes disagree"
            )
        if (
            token_mask.dtype != torch.bool
            or input_ids.dtype not in (torch.int32, torch.int64)
            or previous_token_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise TypeError("Engram expects int32/int64 IDs and a bool token mask")
        return self.input_parameters.hash(input_ids, previous_token_ids, token_mask)

    @property
    def input_parameters(self) -> NGramHashParams:
        """Expose the current immutable buffers to executor-owned input preparation."""
        return NGramHashParams(
            self.token_map,
            self.primes,
            self.reciprocals,
            self.offsets,
            self.multipliers,
            self.pad_id,
        )


def _host_table_dir(nbytes: int) -> str:
    """Pick a writable directory with room for one Engram host-table file.

    ``/dev/shm`` is preferred when it is large enough. Docker often caps it
    at 32–64 GiB, which cannot hold a V4.1 table, so fall back to scratch.
    """
    explicit = global_server_args_dict.get("engram_host_table_dir")
    if explicit:
        return explicit
    margin = 1 << 30
    candidates = []
    if os.path.isdir("/dev/shm"):
        candidates.append("/dev/shm")
    for path in ("/scratch", "/tmp", tempfile.gettempdir()):
        if path not in candidates:
            candidates.append(path)
    for directory in candidates:
        if not os.path.isdir(directory) or not os.access(directory, os.W_OK):
            continue
        if shutil.disk_usage(directory).free >= nbytes + margin:
            return directory
    raise OSError(
        f"No filesystem with {nbytes + margin} free bytes for an Engram host table; "
        "pass --engram-host-table-dir"
    )


_HOST_TABLE_JOB_ID: str | None = None
_CREATED_HOST_FILES: list[str] = []


def _host_table_job_id() -> str:
    """Return a per-job id shared by every rank, unique across concurrent jobs."""
    global _HOST_TABLE_JOB_ID
    if _HOST_TABLE_JOB_ID is not None:
        return _HOST_TABLE_JOB_ID
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        payload = [str(os.getpid())]
        torch.distributed.broadcast_object_list(payload, src=0)
        _HOST_TABLE_JOB_ID = payload[0]
    else:
        _HOST_TABLE_JOB_ID = str(os.getpid())
    return _HOST_TABLE_JOB_ID


def _engram_host_table_path(layer_id: int, field: str, nbytes: int) -> str:
    return os.path.join(
        _host_table_dir(nbytes),
        f"ts-engram-{_host_table_job_id()}-L{layer_id}-{field}",
    )


def _is_host_table_writer(mapping: Mapping) -> bool:
    local = os.environ.get("LOCAL_RANK")
    if local is not None:
        return int(local) == 0
    return mapping.rank % max(mapping.nprocs_per_node, 1) == 0


def resolve_engram_host_layout(host_table: bool, tp_size: int) -> str:
    """Return ``gpu``, ``shared``, or ``sharded`` from the CLI opt-in and TP."""
    if not host_table:
        return "gpu"
    if tp_size < 1:
        raise ValueError("Engram host layout requires a positive attention TP size")
    requested = global_server_args_dict["engram_host_table_layout"]
    if requested == "auto":
        return "sharded" if tp_size > 1 else "shared"
    if requested in ("shared", "sharded"):
        return requested
    raise ValueError(
        "engram_host_table_layout must be auto, shared, or sharded; "
        f"got {requested!r}"
    )


def _cleanup_host_files() -> None:
    while _CREATED_HOST_FILES:
        path = _CREATED_HOST_FILES.pop()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


atexit.register(_cleanup_host_files)


def _advise_hugepages(buffer, nbytes: int) -> None:
    """Best-effort Linux THP hint. Shared file maps often ignore it."""
    if sys.platform != "linux" or nbytes <= 0:
        return
    libc_name = ctypes.util.find_library("c")
    if libc_name is None:
        return
    libc = ctypes.CDLL(libc_name)
    madvise = getattr(libc, "madvise", None)
    if madvise is None:
        return
    madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    madvise.restype = ctypes.c_int
    # Linux MADV_HUGEPAGE=14. Failure is ignored: THP may be disabled.
    madvise(ctypes.c_void_p(int(buffer.ctypes.data)), ctypes.c_size_t(nbytes), 14)


def _create_host_file(path: str, nbytes: int) -> None:
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.ftruncate(fd, nbytes)
    finally:
        os.close(fd)
    _CREATED_HOST_FILES.append(path)


def _unlink_host_file(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    try:
        _CREATED_HOST_FILES.remove(path)
    except ValueError:
        pass


def _host_table_nbytes(shape: tuple[int, int], dtype: torch.dtype) -> int:
    return int(shape[0] * shape[1] * torch.empty((), dtype=dtype).element_size())


def _tensor_from_host_bytes(
    buffer, shape: tuple[int, int], dtype: torch.dtype
) -> torch.Tensor:
    tensor = torch.frombuffer(buffer, dtype=torch.uint8)
    if dtype != torch.uint8:
        tensor = tensor.view(dtype)
    return tensor.view(shape)


def _allocate_host_bytes(
    shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device | str,
    shared_file: bool,
    path: str,
) -> tuple[torch.Tensor, object | None]:
    """Return a host uint8/fp8 table and a keep-alive for the backing store.

    Shared multi-rank tables map a job-scoped file. Rank-local host shards use
    an anonymous buffer so Linux transparent huge pages can apply.
    """
    device = torch.device(device)
    if device.type == "meta":
        return torch.empty(shape, dtype=dtype, device="meta"), None
    nbytes = _host_table_nbytes(shape, dtype)
    if shared_file:
        mmap = np.memmap(path, dtype=np.uint8, mode="r+", shape=(nbytes,))
        _advise_hugepages(mmap, nbytes)
        return _tensor_from_host_bytes(mmap, shape, dtype), mmap
    buffer = np.empty(nbytes, dtype=np.uint8)
    _advise_hugepages(buffer, nbytes)
    return _tensor_from_host_bytes(buffer, shape, dtype), buffer


def _dequant_fp8_e8m0(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    values = codes.view(torch.uint8).view(torch.float8_e4m3fn).float()
    scale = scales.view(torch.uint8).view(torch.float8_e8m0fnu).float()
    return (
        (values.unflatten(-1, (-1, 32)) * scale.unsqueeze(-1))
        .flatten(-2)
        .to(torch.bfloat16)
    )


# TOKENSPEED_ENGRAM_HOST_PREFETCH=1 replaces the UVA row gather of a shared host
# table in pure-prefill forwards. Read once per process; unset or any other
# value keeps the UVA gather everywhere. See EngramHostPrefetcher.
ENGRAM_HOST_PREFETCH_ENV = "TOKENSPEED_ENGRAM_HOST_PREFETCH"
# Host gather threads per process (the helper thread comes on top). Eight
# gather 2048 x 24 rows of the 98 GB table in 0.5 ms; sixteen were slower.
ENGRAM_HOST_PREFETCH_THREADS_ENV = "TOKENSPEED_ENGRAM_HOST_PREFETCH_THREADS"
_PREFETCH_DEFAULT_THREADS = 8
# Below this many rows the UVA gather is cheap and the forward reaches the
# Engram layer before a helper thread could have delivered.
_PREFETCH_MIN_ROWS = 256
# Longest time the forward thread waits for the helper at the Engram layer
# before it runs the UVA gather itself. The helper normally finishes while the
# layers in front of the first Engram layer are still being queued.
_PREFETCH_WAIT_SECONDS = 0.02
_PREFETCH_POLL_SECONDS = 5.0e-5
# Lookups between two reports of the counters.
_PREFETCH_LOG_INTERVAL = 2048
# Back-off. When this many of the last _PREFETCH_MISS_WINDOW lookups were not
# delivered by the helper (late, failed, or staging still owned by an abandoned
# lookup), the following lookups skip the prefetch, so the forward neither
# submits nor waits. A late lookup costs up to _PREFETCH_WAIT_SECONDS and a
# delivered one saves about 3 ms, so a miss rate near one in five is a loss;
# a window, not a run, also catches a first Engram layer that is late in every
# forward while the second one is delivered.
_PREFETCH_MAX_MISSES = 3
_PREFETCH_MISS_WINDOW = 16
# Length of the first pause in lookups. Every further pause is twice as long,
# up to the maximum, until a whole window of lookups was delivered again.
_PREFETCH_PAUSE_LOOKUPS = 256
_PREFETCH_MAX_PAUSE_LOOKUPS = 65536


def engram_host_prefetch_requested() -> bool:
    return os.environ.get(ENGRAM_HOST_PREFETCH_ENV, "") == "1"


def _wait_cuda_event(event: torch.cuda.Event) -> None:
    """Wait for a CUDA event from a helper thread without holding the GIL.

    The calling thread must be bound to the event's device: the query takes
    no device, and a thread that never selected one would use device 0.
    """
    while not event.query():
        time.sleep(_PREFETCH_POLL_SECONDS)


class _EngramPrefetchTable:
    """One host table pair and its pinned staging, owned by one embedding."""

    def __init__(self, codes: torch.Tensor, scales: torch.Tensor):
        # NumPy views of the host tables: np.take releases the GIL and has no
        # thread pool of its own, so the gather threads below set the parallelism.
        self.codes = codes.detach().numpy()
        self.scales = scales.detach().numpy()
        self.codes_ptr = codes.data_ptr()
        self.scales_ptr = scales.data_ptr()
        self.rows = int(codes.shape[0])
        self.capacity = 0
        self.ids: torch.Tensor | None = None
        self.stage_codes: torch.Tensor | None = None
        self.stage_scales: torch.Tensor | None = None
        # NumPy views of the three staging buffers for the gather threads.
        self.host_ids: np.ndarray | None = None
        self.host_codes: np.ndarray | None = None
        self.host_scales: np.ndarray | None = None
        # The last job that used the staging buffers. They are free for the
        # next one once that job's upload has run; only the forward thread
        # reads and writes this field.
        self.owner: _EngramPrefetchJob | None = None

    def reserve(self, count: int, device: torch.device) -> None:
        """Grow the pinned id and staging buffers; only while no job owns them.

        The buffers are page-locked through ``device``'s context, the one the
        readback and the uploads run in. They are created outside inference
        mode whatever mode the calling forward is in: the gather threads fill
        them, and a tensor created in inference mode must not be written from
        a thread that is not in it.
        """
        if count <= self.capacity:
            return
        capacity = max(count, 2 * self.capacity)
        with torch.inference_mode(False), torch.cuda.device(device):
            self.ids = torch.empty(capacity, dtype=torch.int64).pin_memory()
            self.stage_codes = torch.empty(
                (capacity, self.codes.shape[1]), dtype=torch.uint8
            ).pin_memory()
            self.stage_scales = torch.empty(
                (capacity, self.scales.shape[1]), dtype=torch.uint8
            ).pin_memory()
        self.host_ids = self.ids.numpy()
        self.host_codes = self.stage_codes.numpy()
        self.host_scales = self.stage_scales.numpy()
        self.capacity = capacity


class _EngramPrefetchJob:
    """Rows of one Engram lookup on their way from the host table to the device."""

    def __init__(
        self,
        table: _EngramPrefetchTable,
        key: tuple,
        count: int,
        ids_event: torch.cuda.Event,
        codes: torch.Tensor,
        scales: torch.Tensor,
        ids: torch.Tensor,
    ):
        self.table = table
        self.key = key
        self.count = count
        self.ids_event = ids_event
        # Destinations of the upload: written by the helper thread only, read
        # by the forward thread only after ``done_event``.
        self.codes = codes
        self.scales = scales
        # Device copy of the ids the rows are gathered for, made by the forward
        # thread when it queued the readback and never written again. The
        # lookup compares it with its own ids.
        self.ids = ids
        # Recorded on the helper's stream behind the uploads, and set before
        # ``ready``: the job has left the staging and its destinations are
        # complete when this event has happened.
        self.done_event: torch.cuda.Event | None = None
        self.ready = threading.Event()
        self.error: BaseException | None = None


class EngramHostPrefetcher:
    """Host-side gather of Engram rows for pure-prefill forwards.

    The UVA gather kernel pays about 30 ns per random host row, 3 ms of the
    forward stream per Engram layer at 2048 tokens, for 13 MB of data. Here the
    forward thread only queues an asynchronous readback of the row ids when the
    forward starts. A helper thread waits for that readback, gathers codes and
    scales from the host table into pinned staging with a few worker threads,
    and uploads them on its own stream. At the Engram layer the forward stream
    waits for that upload through a CUDA event.

    For every id inside the table the bytes are those the UVA kernel reads; an
    id outside it gathers some row here and row 0 there, and the lookup zeroes
    that row on both paths. The lookup does not rely on the job belonging to
    its forward: it compares a device copy of the ids the rows were gathered
    for with its own and reads every row that differs through UVA, so a job
    from another forward costs time and cannot change the result.

    Threads. The forward thread may run in inference mode, which is
    thread-local; the helper and gather threads never do. They write only the
    pinned staging and the two device destinations of a job, all created
    outside inference mode, and no tensor that the forward thread created in
    its own mode. The forward thread reads a job's destinations only after the
    helper's completion event. The helper and gather threads take the GIL a
    few times per lookup and hold it for microseconds: the ids need no host
    arithmetic, and each gather thread makes two NumPy calls that release it.

    One prefetcher serves one device. Its helper thread binds itself to that
    device and to its own stream before any CUDA call; the gather threads make
    no CUDA calls. The forward thread never synchronizes with the device: it
    waits on the helper for at most ``_PREFETCH_WAIT_SECONDS`` (while the
    layers in front of the Engram layer are still executing) and otherwise
    runs the UVA gather. When ``_PREFETCH_MAX_MISSES`` of the last
    ``_PREFETCH_MISS_WINDOW`` lookups were not delivered, the next
    ``_PREFETCH_PAUSE_LOOKUPS`` lookups skip the prefetch without waiting; the
    pause doubles each time until a whole window is delivered again. Host
    tables are only read; a shared file mapping needs no other handling.
    Pinned memory: 272 bytes per gathered row and table.
    """

    def __init__(self, device: torch.device, threads: int):
        if device.type != "cuda" or device.index is None:
            raise ValueError("The Engram host prefetcher needs an indexed CUDA device")
        self.device = device
        self.stream = torch.cuda.Stream(device)
        self.threads = threads
        self.jobs: queue.SimpleQueue[_EngramPrefetchJob] = queue.SimpleQueue()
        # Helper thread only: jobs whose upload may still be running. They keep
        # the pinned staging and the device destinations alive until it has
        # run, also when the forward and the table have dropped the job.
        self.uploading: list[_EngramPrefetchJob] = []
        self.pool = ThreadPoolExecutor(
            max_workers=threads, thread_name_prefix="engram-gather"
        )
        # Counters. attempts: lookups that asked for a prefetch; used: served
        # by the helper; late and failed: the forward ran the UVA gather after
        # waiting; busy: staging still owned by an abandoned lookup; skipped:
        # not submitted during a pause.
        self.attempts = 0
        self.submitted = 0
        self.used = 0
        self.late = 0
        self.failed = 0
        self.busy = 0
        self.skipped = 0
        self.pauses = 0
        self.waited_seconds = 0.0
        self.gather_seconds = 0.0
        # Back-off state, forward thread only: 1 for every recent lookup the
        # helper did not deliver, 0 for a delivered one.
        self.recent: collections.deque[int] = collections.deque(
            maxlen=_PREFETCH_MISS_WINDOW
        )
        self.pause_left = 0
        self.next_pause = _PREFETCH_PAUSE_LOOKUPS
        self.warned_late = False
        self.warned_failed = False
        self.thread = threading.Thread(
            target=self._run, name=f"engram-prefetch-{device.index:d}", daemon=True
        )
        self.thread.start()

    def submit(
        self,
        table: _EngramPrefetchTable,
        indices: torch.Tensor,
        key: tuple,
    ) -> _EngramPrefetchJob | None:
        """Queue the gather of the lookup's rows; forward thread, no waiting.

        Args:
            table: Host table pair of the calling embedding; this rank holds
                every row of it from row 0.
            indices: Int64 ids ``[rows, hash columns]`` of the lookup, on this
                prefetcher's device.
            key: Identity of the lookup, checked again when it is consumed.

        Returns:
            The job to pass to ``take``, or None when the lookup must use the
            UVA gather: during a back-off pause, when the helper thread has
            stopped, or when the table's staging is still owned by an earlier
            job that the forward abandoned.
        """
        self.attempts += 1
        if self.attempts % _PREFETCH_LOG_INTERVAL == 0:
            logger.info(f"Engram host prefetch on {self.device}: {self._counters()}")
        if self.pause_left > 0:
            self.pause_left -= 1
            self.skipped += 1
            return None
        if not self.thread.is_alive():
            self.failed += 1
            self._miss("failed", "the helper thread has stopped")
            return None
        if table.owner is not None and not self._finished(table.owner):
            self.busy += 1
            self._miss("late", "the staging is still owned by an earlier lookup")
            return None
        count = indices.numel()
        device = self.device
        table.reserve(count, device)
        # Allocate the destinations before the id readback is queued: once the
        # helper has seen the readback event, every earlier use of this device
        # memory on the forward stream has run, so the upload on the helper's
        # stream cannot overtake it. The helper thread writes them, so they
        # are ordinary tensors also when this forward runs in inference mode:
        # a tensor created in inference mode may only be written in that mode,
        # which is a property of the forward thread alone.
        with torch.inference_mode(False):
            codes = torch.empty(
                (*indices.shape, table.codes.shape[1]),
                dtype=torch.uint8,
                device=device,
            )
            scales = torch.empty(
                (*indices.shape, table.scales.shape[1]),
                dtype=torch.uint8,
                device=device,
            )
        # The ids as they are now, contiguous: the rows are gathered for this
        # copy, which only the forward thread ever touches, and the lookup
        # compares it with the ids it is called with.
        ids = indices.clone()
        # Asynchronous readback of that copy into the pinned staging.
        table.ids[:count].view(indices.shape).copy_(ids, non_blocking=True)
        ids_event = torch.cuda.Event()
        ids_event.record(torch.cuda.current_stream(device))
        job = _EngramPrefetchJob(table, key, count, ids_event, codes, scales, ids)
        table.owner = job
        self.submitted += 1
        self.jobs.put(job)
        return job

    def _finished(self, job: _EngramPrefetchJob) -> bool:
        """Forward thread: has the helper left this job and has its upload run?"""
        if not job.ready.is_set() or job.done_event is None:
            return False
        # An event query takes no device; make sure it runs in this one's context.
        if torch.cuda.current_device() == self.device.index:
            return job.done_event.query()
        with torch.cuda.device(self.device):
            return job.done_event.query()

    def take(
        self, job: _EngramPrefetchJob
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return device codes, scales and the ids they belong to, or None.

        The forward thread waits here for the helper thread only, never for the
        device; the upload is ordered in front of the caller's next kernels by
        a stream wait on its completion event. The caller must check the ids
        against its own before it uses a row.
        """
        if not job.ready.is_set():
            started = time.perf_counter()
            arrived = job.ready.wait(_PREFETCH_WAIT_SECONDS)
            self.waited_seconds += time.perf_counter() - started
            if not arrived:
                self.late += 1
                self._miss("late", "the host gather was not ready in time")
                return None
        if job.error is not None:
            self.failed += 1
            self._miss("failed", f"the host gather failed: {job.error!r}")
            return None
        torch.cuda.current_stream(self.device).wait_event(job.done_event)
        self.used += 1
        self.recent.append(0)
        if len(self.recent) == _PREFETCH_MISS_WINDOW and not any(self.recent):
            self.next_pause = _PREFETCH_PAUSE_LOOKUPS
        if self.used == 1:
            logger.info(
                f"Engram host prefetch on {self.device} is active: {self._counters()}"
            )
        return job.codes, job.scales, job.ids

    def _miss(self, kind: str, reason: str) -> None:
        """Count an undelivered lookup, warn once per kind and start a pause."""
        if kind == "late" and not self.warned_late:
            self.warned_late = True
            logger.warning(
                f"Engram host prefetch on {self.device} used the UVA gather: "
                f"{reason} (first late lookup; later ones are only counted)"
            )
        if kind == "failed" and not self.warned_failed:
            self.warned_failed = True
            logger.warning(
                f"Engram host prefetch on {self.device} used the UVA gather: "
                f"{reason} (first failed lookup; later ones are only counted)"
            )
        self.recent.append(1)
        if sum(self.recent) < _PREFETCH_MAX_MISSES:
            return
        window = len(self.recent)
        self.recent.clear()
        self.pauses += 1
        self.pause_left = self.next_pause
        logger.warning(
            f"Engram host prefetch on {self.device} backs off: "
            f"{_PREFETCH_MAX_MISSES:d} of the last {window:d} lookups were not delivered "
            f"(last: {reason}); the next {self.pause_left:d} lookups use the UVA "
            f"gather without waiting (pause {self.pauses:d}). {self._counters()}"
        )
        self.next_pause = min(2 * self.next_pause, _PREFETCH_MAX_PAUSE_LOOKUPS)

    def _counters(self) -> str:
        return (
            f"{self.attempts:d} lookups, used {self.used:d}, late {self.late:d}, "
            f"failed {self.failed:d}, busy {self.busy:d}, skipped in pauses "
            f"{self.skipped:d}, pauses {self.pauses:d}, forward waited "
            f"{1e3 * self.waited_seconds:.1f} ms in total, host gather "
            f"{1e3 * self.gather_seconds / max(self.submitted, 1):.2f} ms per lookup "
            f"({self.threads:d} threads)"
        )

    def _run(self) -> None:
        # A new thread starts on CUDA device 0. Bind it to this prefetcher's
        # device and stream first: every later call (event queries, uploads)
        # then stays in the rank's own context, and no context is created on
        # a GPU that belongs to another rank.
        torch.cuda.set_device(self.device)
        torch.cuda.set_stream(self.stream)
        while True:
            self._serve(self.jobs.get())

    def _serve(self, job: _EngramPrefetchJob) -> None:
        # Drop the jobs whose upload has run; the new one stays referenced
        # here until a later call finds its upload finished.
        self.uploading = [
            old
            for old in self.uploading
            if old.done_event is None or not old.done_event.query()
        ]
        self.uploading.append(job)
        try:
            self._process(job)
        except BaseException as error:  # noqa: BLE001 - reported by take()
            job.error = error
            try:
                # An upload queued before the failure still reads the staging
                # and writes the job's destinations: the job is finished when
                # this thread's stream has run everything queued on it so far.
                done = torch.cuda.Event()
                done.record(self.stream)
                job.done_event = done
            finally:
                job.ready.set()

    @staticmethod
    def _take_rows(table: _EngramPrefetchTable, start: int, stop: int) -> None:
        # Host memory only: the gather threads make no CUDA and no torch call.
        # "clip" keeps an id outside the table inside it; the lookup zeroes the
        # row of such an id whatever was gathered for it.
        ids = table.host_ids[start:stop]
        np.take(
            table.codes, ids, axis=0, out=table.host_codes[start:stop], mode="clip"
        )
        np.take(
            table.scales, ids, axis=0, out=table.host_scales[start:stop], mode="clip"
        )

    def _process(self, job: _EngramPrefetchJob) -> None:
        table = job.table
        count = job.count
        # The ids have reached the host, and everything queued on the forward
        # stream before them has run.
        _wait_cuda_event(job.ids_event)
        started = time.perf_counter()
        step = max(1024, -(-count // self.threads))
        pending = [
            self.pool.submit(self._take_rows, table, start, min(start + step, count))
            for start in range(0, count, step)
        ]
        # One wait for all gather threads, then their errors if any.
        wait_futures(pending)
        for future in pending:
            future.result()
        self.gather_seconds += time.perf_counter() - started
        # Uploads run on this thread's current stream, which _run set to the
        # prefetcher's own stream on its device.
        job.codes.view(count, -1).copy_(table.stage_codes[:count], non_blocking=True)
        job.scales.view(count, -1).copy_(table.stage_scales[:count], non_blocking=True)
        done = torch.cuda.Event()
        done.record(self.stream)
        job.done_event = done
        # The helper does not wait for the upload: the staging stays with this
        # job until the forward thread sees ``done`` (submit), and the job
        # stays referenced by ``uploading``.
        job.ready.set()

    def stats(self) -> dict[str, float]:
        """Counters since start, for logs and benchmarks."""
        return {
            "attempts": self.attempts,
            "submitted": self.submitted,
            "used": self.used,
            "late": self.late,
            "failed": self.failed,
            "busy": self.busy,
            "skipped": self.skipped,
            "pauses": self.pauses,
            "pause_left": self.pause_left,
            "waited_seconds": self.waited_seconds,
            "gather_seconds": self.gather_seconds,
        }


_ENGRAM_HOST_PREFETCHERS: dict[torch.device, EngramHostPrefetcher] = {}


def engram_host_prefetcher(device: torch.device) -> EngramHostPrefetcher:
    """Return the process-wide prefetcher of a device, starting it on first use."""
    prefetcher = _ENGRAM_HOST_PREFETCHERS.get(device)
    if prefetcher is None:
        threads = int(
            os.environ.get(ENGRAM_HOST_PREFETCH_THREADS_ENV, _PREFETCH_DEFAULT_THREADS)
        )
        if threads < 1:
            raise ValueError(f"{ENGRAM_HOST_PREFETCH_THREADS_ENV} must be positive")
        prefetcher = EngramHostPrefetcher(device, threads)
        _ENGRAM_HOST_PREFETCHERS[device] = prefetcher
    return prefetcher


def _engram_lookup_key(indices: torch.Tensor) -> tuple:
    # A cheap filter only. The hash ids live in a pointer-stable input buffer,
    # so two forwards with the same row count have the same key; the lookup
    # therefore verifies every prefetched row against its ids on the device.
    return (indices.data_ptr(), tuple(indices.shape), indices.stride(), indices.device)


class RowShardedEngramEmbedding(nn.Module):
    """FP8 codes and raw E8M0 scale bytes, GPU-sharded or host-resident."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mapping: Mapping,
        device: torch.device | str,
        host_table: bool,
        host_layout: str,
        layer_id: int,
    ):
        super().__init__()
        if (
            num_embeddings < mapping.attn.tp_size
            or embedding_dim < 32
            or embedding_dim % 32
        ):
            raise ValueError(
                "Engram embedding requires rows per TP rank and 32-wide scale blocks"
            )
        if host_table:
            if host_layout not in ("shared", "sharded"):
                raise ValueError(
                    "Host Engram tables require host_layout shared or sharded"
                )
        elif host_layout != "gpu":
            raise ValueError("GPU Engram tables require host_layout='gpu'")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_group = mapping.attn.tp_group
        self.host_table = host_table
        self.host_layout = host_layout
        self.layer_id = layer_id
        self._mmap_keep: list[object] = []
        self._host_gpu_registered = False
        # TOKENSPEED_ENGRAM_HOST_PREFETCH: the pending host gather of this
        # forward, and the host-table views it reads (built on first use).
        self._host_prefetch = (
            host_table and host_layout == "shared" and engram_host_prefetch_requested()
        )
        self._prefetch_table: _EngramPrefetchTable | None = None
        self._prefetch_job: _EngramPrefetchJob | None = None
        row_sharded = (not host_table) or host_layout == "sharded"
        if row_sharded:
            self.part_num_embeddings = (
                num_embeddings + mapping.attn.tp_size - 1
            ) // mapping.attn.tp_size
            self.row_start = mapping.attn.tp_rank * self.part_num_embeddings
            self.row_end = min(
                self.row_start + self.part_num_embeddings, num_embeddings
            )
        else:
            self.part_num_embeddings = num_embeddings
            self.row_start = 0
            self.row_end = num_embeddings
        shared_file = (
            host_layout == "shared"
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        self._shared_writer = (not shared_file) or _is_host_table_writer(mapping)
        if host_table:
            weight_path = ""
            scale_path = ""
            weight_shape = (self.part_num_embeddings, embedding_dim)
            scale_shape = (self.part_num_embeddings, embedding_dim // 32)
            if shared_file:
                weight_bytes = _host_table_nbytes(weight_shape, torch.float8_e4m3fn)
                scale_bytes = _host_table_nbytes(scale_shape, torch.uint8)
                weight_path = _engram_host_table_path(layer_id, "weight", weight_bytes)
                scale_path = _engram_host_table_path(layer_id, "scale", scale_bytes)
                if self._shared_writer:
                    _create_host_file(weight_path, weight_bytes)
                    _create_host_file(scale_path, scale_bytes)
                torch.distributed.barrier()
            weight, weight_mmap = _allocate_host_bytes(
                weight_shape,
                torch.float8_e4m3fn,
                device,
                shared_file,
                weight_path,
            )
            scale, scale_mmap = _allocate_host_bytes(
                scale_shape,
                torch.uint8,
                device,
                shared_file,
                scale_path,
            )
            for keep in (weight_mmap, scale_mmap):
                if keep is not None:
                    self._mmap_keep.append(keep)
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.scale = nn.Parameter(scale, requires_grad=False)
            if shared_file:
                torch.distributed.barrier()
                if self._shared_writer:
                    _unlink_host_file(weight_path)
                    _unlink_host_file(scale_path)
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    self.part_num_embeddings,
                    embedding_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
                requires_grad=False,
            )
            self.scale = nn.Parameter(
                torch.empty(
                    self.part_num_embeddings,
                    embedding_dim // 32,
                    dtype=torch.uint8,
                    device=device,
                ),
                requires_grad=False,
            )
        for param in (self.weight, self.scale):
            param.weight_loader = self.weight_loader
            param.engram_row_sharded = row_sharded
            param.engram_host_table = host_table
            param.engram_row_start = self.row_start
            param.engram_row_end = self.row_end
        local_rows = self.row_end - self.row_start
        if torch.device(device).type != "meta":
            self.weight.data[local_rows:].zero_()
            self.scale.data[local_rows:].fill_(127)
            self._register_host_for_gpu()

    def _register_host_for_gpu(self) -> None:
        """cudaHostRegister once. Gather kernels must not register."""
        if self._host_gpu_registered or not self.host_table:
            return
        if (
            torch.device(self.weight.device).type != "cpu"
            or not torch.cuda.is_available()
        ):
            self._host_gpu_registered = True
            return
        platform = current_platform()
        for param in (self.weight, self.scale):
            # Keep storage alive without retaining the parameter's bound loader.
            tensor = param.detach()
            if not tensor.is_pinned():
                platform.register_host_tensor_for_gpu_access(tensor)
                weakref.finalize(
                    self, platform.unregister_host_tensor_for_gpu_access, tensor
                )
        self._host_gpu_registered = True

    def _apply(self, fn, recurse=True):
        if self.host_table:
            return self
        return super()._apply(fn, recurse)

    def _copy_rows(
        self, param: nn.Parameter, rows: torch.Tensor, local_start: int
    ) -> None:
        if param is self.scale:
            if rows.dtype not in (torch.uint8, torch.float8_e8m0fnu):
                raise TypeError(
                    "Engram scales must be E8M0 or raw uint8 exponent bytes"
                )
            rows = rows.view(torch.uint8)
        elif rows.dtype != torch.float8_e4m3fn:
            raise TypeError("Engram table weights must stay FP8 E4M3")
        param.data[local_start : local_start + rows.shape[0]].copy_(rows)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Load CPU full-table views or already-local tensors without dequantization.

        The streaming loader should prefer load_sharded(): this callback cannot
        undo a full-table allocation already made by a generic weight iterator.
        """
        if param is not self.weight and param is not self.scale:
            raise ValueError("Parameter does not belong to this Engram table")
        if loaded_weight.ndim != 2 or loaded_weight.shape[1] != param.shape[1]:
            raise ValueError("Engram table weight has the wrong row width")
        rows = loaded_weight.shape[0]
        local_rows = self.row_end - self.row_start
        if rows == self.num_embeddings:
            if loaded_weight.device.type != "cpu" and len(self.tp_group) > 1:
                raise ValueError(
                    "Do not materialize a full Engram table on GPU; use load_sharded"
                )
            loaded_weight = loaded_weight[self.row_start : self.row_end]
        elif rows in (local_rows, self.part_num_embeddings):
            loaded_weight = loaded_weight[:local_rows]
        else:
            raise ValueError(
                "Engram weight is neither a full table nor this rank's row shard"
            )
        self._copy_rows(param, loaded_weight, 0)

    def load_sharded(self, parameter_name: str, tensor_slice, chunk_rows: int) -> None:
        """Copy table rows from a CPU safetensors get_slice() in bounded chunks.

        GPU and host-sharded tables copy only this rank's rows. Shared host
        tables copy every row on the writer rank; followers map the file.
        Args: parameter_name is 'weight' or 'scale'; tensor_slice is the global
        checkpoint tensor's lazy CPU slice; chunk_rows bounds transient host
        storage. No full tensor, remote shard, or BF16 table is allocated.
        Returns: None; the caller records the corresponding parameter as loaded.
        """
        if parameter_name not in ("weight", "scale") or chunk_rows < 1:
            raise ValueError("Expected weight/scale and a positive row chunk size")
        if self.host_table and not self._shared_writer:
            return
        param = getattr(self, parameter_name)
        if tuple(tensor_slice.get_shape()) != (self.num_embeddings, param.shape[1]):
            raise ValueError("Checkpoint Engram shape does not match the global table")
        for start in range(self.row_start, self.row_end, chunk_rows):
            end = min(start + chunk_rows, self.row_end)
            rows = tensor_slice[start:end, :]
            if rows.device.type != "cpu":
                raise ValueError("Open Engram safetensors on CPU before slicing")
            self._copy_rows(param, rows, start - self.row_start)

    def prefetch(self, indices: torch.Tensor) -> None:
        """Start the host gather for the ``forward(indices)`` of this forward.

        Call it when a pure-prefill, eager forward starts, with the tensor the
        later ``forward`` receives. It never waits. It is a no-op unless
        TOKENSPEED_ENGRAM_HOST_PREFETCH=1 selected it for a shared host table
        (this rank then holds every row, from row 0), the ids are a device
        [rows, hash columns] tensor with enough rows, the prefetcher is not in
        a back-off pause and no earlier job still owns this table's staging.
        """
        self._prefetch_job = None
        if (
            not self._host_prefetch
            or self.row_start != 0
            or not indices.is_cuda
            or indices.ndim != 2
            or indices.dtype != torch.int64
            or indices.shape[0] < _PREFETCH_MIN_ROWS
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        codes = self.weight.detach().view(torch.uint8)
        table = self._prefetch_table
        if (
            table is None
            or table.codes_ptr != codes.data_ptr()
            or table.scales_ptr != self.scale.data_ptr()
        ):
            table = _EngramPrefetchTable(codes, self.scale)
            self._prefetch_table = table
        self._prefetch_job = engram_host_prefetcher(indices.device).submit(
            table, indices, _engram_lookup_key(indices)
        )

    def _take_prefetched(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        job = self._prefetch_job
        self._prefetch_job = None
        if (
            job is None
            or job.key != _engram_lookup_key(indices)
            # A captured lookup must not hold a prefetched buffer as a constant.
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        return engram_host_prefetcher(indices.device).take(job)

    def forward(
        self, indices: torch.Tensor, *, query_shard: QueryShardPlan | None
    ) -> torch.Tensor:
        """Gather [..., hash columns] IDs into BF16 [..., hash columns, head dim].

        Without a query shard, TP peers pass identical indices. A query shard
        gathers IDs and reduce-scatters table contributions to its local rows.
        Shared host tables gather the full copy and skip the all-reduce. GPU
        and host-sharded tables expand only local rows and reduce.
        Dequantization uses the same FP8/E8M0 torch casts in both paths.
        """
        sharded = query_shard is not None and query_shard.size > 1
        if sharded:
            if indices.ndim != 2 or indices.shape[0] != query_shard.local_rows:
                raise ValueError("Engram hashes must match the query shard rows")
            if len(self.tp_group) != query_shard.size:
                raise ValueError(
                    "Engram TP and query shard groups must have equal size"
                )
            if self.host_layout != "shared":
                indices = token_all_gather_rows(
                    indices, self.tp_group, list(query_shard.row_counts)
                )
        if indices.is_cuda and not self.host_table:
            from tokenspeed_kernel.ops.embedding import mxfp8_embedding

            # Each layer's 2D hash slice can be gathered directly; retain the
            # contiguous fallback for arbitrary leading dimensions.
            if indices.ndim != 2:
                indices = indices.contiguous()
            values = mxfp8_embedding(
                self.weight,
                self.scale,
                indices,
                self.row_start,
                self.row_end,
            )
        else:
            local = (indices >= self.row_start) & (indices < self.row_end)
            local_ids = (indices - self.row_start).masked_fill(~local, 0)
            if self.host_table:
                prefetched = self._take_prefetched(indices)
                if prefetched is None:
                    codes = host_gather.uint8_row_gather(
                        self.weight.view(torch.uint8), local_ids, None
                    )
                    scales = host_gather.uint8_row_gather(self.scale, local_ids, None)
                else:
                    # Same bytes, gathered on the host and uploaded once. A row
                    # is right only if it was gathered for this lookup's id, and
                    # the job cannot prove which forward it belongs to, so every
                    # row is checked here. Rows whose id differs (none, unless
                    # the job is from another forward) are read through UVA; an
                    # id of -1 makes that kernel write zeros without a host
                    # read, so the check costs two launches that read nothing.
                    # A row whose id is outside the table is zeroed below.
                    codes, scales, gathered_ids = prefetched
                    stale = gathered_ids != indices
                    again = local_ids.masked_fill(~stale, -1)
                    stale = stale.unsqueeze(-1)
                    codes = torch.where(
                        stale,
                        host_gather.uint8_row_gather(
                            self.weight.view(torch.uint8), again, None
                        ),
                        codes,
                    )
                    scales = torch.where(
                        stale,
                        host_gather.uint8_row_gather(self.scale, again, None),
                        scales,
                    )
                values = _dequant_fp8_e8m0(codes, scales)
            else:
                local_ids = local_ids.long()
                # Byte gathers work on CPU and CUDA, including dtypes without index kernels.
                values = _dequant_fp8_e8m0(
                    self.weight.view(torch.uint8)[local_ids], self.scale[local_ids]
                )
            values = values.masked_fill(~local.unsqueeze(-1), 0)
        if len(self.tp_group) > 1 and self.host_layout != "shared":
            # The workspace all-reduce sizes itself on rows x trailing width.
            # Reduce one row per token, [rows, columns * head_dim]: the lane is
            # widened for that at construction (DeepseekV41Engram), so a
            # decode batch stays inside the one-shot window. As
            # [rows * columns, head_dim] the same bytes exceed the window's
            # row capacity and drop to NCCL, which costs a ring latency plus a
            # launch stall on every step.
            flat = values.reshape(-1, values.shape[-2] * values.shape[-1])
            if sharded:
                return token_reduce_scatter(
                    flat, self.tp_group, list(query_shard.row_counts)
                ).reshape(query_shard.local_rows, values.shape[-2], values.shape[-1])
            values = all_reduce(
                flat,
                group=self.tp_group,
                backend=None,
                op=torch.distributed.ReduceOp.SUM,
            ).view(values.shape)
        return values


def engram_reduce_lane_width(layout: EngramLayout) -> int:
    """Trailing width of the embedding all-reduce: one row per token.

    The gathered values are ``[tokens, n_hash_cols, head_dim]``; they are
    reduced as ``[tokens, n_hash_cols * head_dim]`` so the workspace
    all-reduce sees one row per token and a decode batch stays inside its
    one-shot row window. This is the lane width to arm for it.
    """
    return (layout.max_ngram_size - 1) * layout.n_heads * layout.head_dim


class DeepseekV41Engram(nn.Module):
    """Reference normalized-dot, signed-sqrt gate over four hyperconnection copies."""

    def __init__(
        self,
        config,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
        device: torch.device | str,
        host_table: bool,
        host_layout: str,
    ):
        """Construct with the V4.1 runtime quant config selected by the caller.

        prefix is the runtime module path (e.g. model.layers.1.engram). The
        checkpoint's wkv is [hidden_size * (hc_mult + 1), n_hash_cols * head_dim],
        replicated and quantized through the existing linear implementation.
        """
        super().__init__()
        layout = EngramLayout.from_config(config)
        if config.hc_mult != 4:
            raise ValueError("V4.1 Engram requires four hyperconnection copies")
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.prefix = prefix
        self.n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        if quant_config is not None:
            if (
                not isinstance(quant_config, Mxfp8Config)
                or not quant_config.is_checkpoint_fp8_serialized
                or quant_config.weight_block_size != [1, 32]
                or quant_config.scale_fmt != "ue8m0"
            ):
                raise ValueError(
                    "Engram wkv requires the V4.1 runtime 1x32 MXFP8 config"
                )
        with torch.device(device):
            self.wkv = ReplicatedLinear(
                input_size=self.n_hash_cols * layout.head_dim,
                output_size=self.dim * (self.hc_mult + 1),
                bias=False,
                skip_bias_add=False,
                params_dtype=torch.bfloat16,
                quant_config=quant_config,
                prefix=f"{prefix}.wkv",
            )
            self.q_weight = nn.Parameter(
                torch.ones(self.hc_mult, self.dim, dtype=torch.bfloat16),
                requires_grad=False,
            )
            self.k_weight = nn.Parameter(
                torch.ones(self.hc_mult, self.dim, dtype=torch.bfloat16),
                requires_grad=False,
            )
        if quant_config is not None and not isinstance(
            self.wkv.quant_method, Fp8LinearMethod
        ):
            raise ValueError(
                "Do not exclude Engram wkv from checkpoint FP8 quantization"
            )
        self.embed = RowShardedEngramEmbedding(
            layout.num_embeddings[self.layer_hash_index],
            layout.head_dim,
            mapping,
            device,
            host_table,
            host_layout,
            layer_id,
        )
        self.q_weight.weight_loader = default_weight_loader
        self.k_weight.weight_loader = default_weight_loader
        # The embedding reduces [tokens, n_hash_cols * head_dim] across attention
        # TP; widen the one-shot lane to that width so decode batches take the
        # workspace kernel. Collective: every rank builds the same layers.
        self.reduce_lane_armed = False
        if (
            len(mapping.attn.tp_group) > 1
            and host_layout != "shared"
            and torch.distributed.is_initialized()
        ):
            width = engram_reduce_lane_width(layout)
            self.reduce_lane_armed = prepare_all_reduce_lane(
                mapping.attn.tp_group, width
            )
            if not self.reduce_lane_armed:
                logger.warning(
                    f"{prefix!s}: one-shot all-reduce lane of width {width:d} not "
                    "armed; the Engram embedding reduce falls back to NCCL"
                )

    def checkpoint_weight_aliases(self) -> dict[str, str]:
        """Return raw layers.<id>.engram checkpoint names to runtime param paths.

        Use these before the generic '.scale' rename. Embedding params expose
        engram_row_sharded/row_start/row_end attributes for streaming dispatch.
        """
        names = {
            "embed.weight": "embed.weight",
            "embed.scale": "embed.scale",
            "wkv.weight": "wkv.weight",
            "q_weight": "q_weight",
            "k_weight": "k_weight",
        }
        if hasattr(self.wkv, "weight_scale_inv"):
            names["wkv.scale"] = "wkv.weight_scale_inv"
        return {
            f"layers.{self.layer_id}.engram.{source}": f"{self.prefix}.{target}"
            for source, target in names.items()
        }

    def prefetch(self, hash_ids: torch.Tensor) -> None:
        """Start this layer's host-table gather early; see the embedding."""
        self.embed.prefetch(hash_ids)

    def lookup_is_local(self) -> bool:
        """Whether ``lookup`` runs without a collective on this rank."""
        embed = self.embed
        return len(embed.tp_group) <= 1 or embed.host_layout == "shared"

    def lookup(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """The embeddings ``forward`` gathers for ``hash_ids``, without a shard.

        They depend on the hash ids only, so a caller may compute them ahead
        of the layer (on another stream) and hand them to ``forward`` as
        ``embeddings``: the same kernels on the same inputs, bit-identical.
        Requires ``lookup_is_local()``.
        """
        return self.embed(hash_ids, query_shard=None).flatten(-2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        query_shard: QueryShardPlan | None,
        embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inject Engram into [..., 4, hidden_size], returning the same shape/dtype.

        hash_ids contains this layer's [..., n_hash_cols] IDs; token_mask is bool
        [...]. False positions pass through unchanged, including image/pad rows.
        Norms and gates accumulate in FP32, independently for each HC copy.
        """
        shape = hidden_states.shape[:-2]
        if (
            hidden_states.shape[-2:] != (self.hc_mult, self.dim)
            or hash_ids.shape != (*shape, self.n_hash_cols)
            or token_mask.shape != shape
        ):
            raise ValueError("Engram hidden states, hashes and mask shapes disagree")
        if token_mask.dtype != torch.bool:
            raise TypeError("Engram token mask must be bool")
        if hidden_states.numel() == 0 and query_shard is None:
            return hidden_states
        if embeddings is None:
            embeddings = self.embed(hash_ids, query_shard=query_shard).flatten(-2)
        elif query_shard is not None or embeddings.shape != (
            *shape,
            self.wkv.input_size,
        ):
            raise ValueError("Engram embeddings computed ahead must match the rows")
        if hidden_states.numel() == 0:
            return hidden_states
        kv, _ = self.wkv(embeddings, block_scale=None, output_dtype=None)
        if hidden_states.is_cuda:
            from tokenspeed_kernel.ops.residual import normalized_dot_gate

            return normalized_dot_gate(
                hidden_states, kv, self.q_weight, self.k_weight, token_mask, self.eps
            )
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        h = hidden_states.float()
        weight = self.q_weight.float() * self.k_weight.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(
            hidden_states.dtype
        )
