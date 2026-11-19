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

"""A per-layer expert placement read from a file, and a per-layer load profile.

Two switches for expert-parallel MoE whose kernel owns the exchange (the
SM120 MegaMoE), both off unless their environment variable is set:

``TOKENSPEED_MOE_EXPERT_PLACEMENT=<file>``
    Each layer's experts are permuted over the EP ranks. Slot ``p`` of layer
    ``L`` holds logical expert ``physical_to_logical[L][p]``; rank ``r`` of an
    EP group of size ``E`` owns slots ``[r * n / E, (r + 1) * n / E)``. The
    router keeps logical numbering; routed ids are mapped to slots where the
    kernel inputs are staged, so the routes, the per-route results and the
    order a token's routes are summed in are those of contiguous ownership.

``TOKENSPEED_MOE_LOAD_PROFILE=<directory>``
    Every rank counts, per MoE layer, the routes its own rows send to each
    logical expert and writes the totals to ``<directory>/rank<r>.json``.
    The file is written from eager forwards only. A process that serves from
    CUDA graphs keeps counting on the GPU but runs no Python in the layer and
    is killed without exit handlers, so its file stays at the last eager
    forward, usually the start-up warm-up; the file's ``coverage`` says so.

The placement file::

    {"format": "tokenspeed-moe-expert-placement/1",
     "num_experts": 384, "num_layers": 40, "ep_sizes": [8],
     "physical_to_logical": [[<384 expert ids>], ... one row per layer]}

``ep_sizes`` lists the EP group sizes the file was solved for; a model whose
EP size is not listed refuses the file.

Format 2 adds replicas: more slots than experts, so a hot expert can sit on
every rank and each rank routes its own tokens' hits on it to its own copy::

    {"format": "tokenspeed-moe-expert-placement/2",
     "num_experts": 384, "num_physical_experts": 400,
     "num_layers": 40, "ep_sizes": [8],
     "physical_to_logical": [[<400 expert ids>], ...],
     "home_slot": [[<384 slots>], ...]}

A slot's entry is the logical expert whose weights it holds; an expert may
fill several slots (replicas, and fillers that no route reaches). A rank
routes logical expert ``e`` to its own first slot holding ``e`` when it has
one, otherwise to ``home_slot[L][e]``. The MoE layers are then built with
``num_physical_experts`` slots, the same number on every rank.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

EXPERT_PLACEMENT_ENV = "TOKENSPEED_MOE_EXPERT_PLACEMENT"
LOAD_PROFILE_ENV = "TOKENSPEED_MOE_LOAD_PROFILE"
# Forwards between two writes of the profile (default 200).
LOAD_PROFILE_INTERVAL_ENV = "TOKENSPEED_MOE_LOAD_PROFILE_INTERVAL"
PLACEMENT_FORMAT = "tokenspeed-moe-expert-placement/1"
PLACEMENT_FORMAT_REPLICAS = "tokenspeed-moe-expert-placement/2"
# The fused kernel holds at most this many expert slots per rank.
_MAX_SLOTS_PER_RANK = 128
PROFILE_FORMAT = "tokenspeed-moe-load-profile/1"
# Forward kinds the counters keep apart, in the order of the counter rows.
LOAD_MODES = ("extend", "decode", "mixed", "other")
_DEFAULT_PROFILE_INTERVAL = 200
# A profile is also written when this many seconds passed since the last one.
_PROFILE_MAX_AGE_S = 60.0


@dataclass(frozen=True)
class FileExpertPlacement:
    """One validated placement file.

    Carries the members ``build_moe_checkpoint_loader`` reads of a placement,
    so the checkpoint loader fills each rank's slots from it.
    """

    path: str
    sha256: str
    ep_size: int
    num_layers: int
    num_logical_experts: int
    # Expert slots over all ranks; more than the logical experts when the
    # file replicates some.
    num_physical_experts: int
    # [layer][slot] -> logical expert whose weights the slot holds.
    physical_to_logical: tuple[tuple[int, ...], ...]
    # [layer][logical expert] -> the slot a rank routes to when it holds no
    # copy of the expert itself.
    logical_to_physical: tuple[tuple[int, ...], ...]

    @property
    def digest(self) -> int:
        """A positive 62-bit digest of the file, to compare across ranks."""
        return _digest(self.sha256)

    def local_slot_logical_experts(self, layer_id: int, ep_rank: int) -> list[int]:
        """The logical expert in each of ``ep_rank``'s slots of ``layer_id``."""
        local = self.num_physical_experts // self.ep_size
        return list(
            self.physical_to_logical[layer_id][ep_rank * local : (ep_rank + 1) * local]
        )

    def rank_logical_to_physical(self, layer_id: int, ep_rank: int) -> list[int]:
        """Slot that ``ep_rank`` routes each logical expert of ``layer_id`` to.

        Its own first slot holding the expert when it has one (a replica, or
        the expert's only slot), otherwise the expert's home slot on another
        rank.
        """
        local = self.num_physical_experts // self.ep_size
        slots = list(self.logical_to_physical[layer_id])
        own = self.local_slot_logical_experts(layer_id, ep_rank)
        for index in range(local - 1, -1, -1):
            slots[own[index]] = ep_rank * local + index
        return slots

    def id_map(self, layer_id: int, ep_rank: int, device: torch.device) -> torch.Tensor:
        """int32 ``[num_experts]``: logical expert id -> slot, as ``ep_rank`` routes."""
        return torch.tensor(
            self.rank_logical_to_physical(layer_id, ep_rank),
            dtype=torch.int32,
            device=device,
        )


def _fail(path: str, reason: str) -> ValueError:
    return ValueError(f"{EXPERT_PLACEMENT_ENV}={path}: {reason}")


def _digest(sha256: str) -> int:
    """A positive 62-bit digest of a file's SHA-256, to compare across ranks."""
    return int(sha256[:16], 16) >> 2 | 1


@dataclass(frozen=True)
class PlacementFileIdentity:
    """What a process knows of the environment's placement file before a
    model validates it: enough to size the MoE layers and to compare the
    file across the EP ranks."""

    path: str
    # Digest of the file's bytes; equals ``FileExpertPlacement.digest`` of the
    # placement a model loads from the same bytes.
    digest: int
    # 1 (permutation), 2 (replicas), 0 when the file is no placement file.
    format: int
    # The file's ``num_experts`` and ``num_physical_experts`` as written
    # (None when absent); a model validates them when it loads the file.
    num_experts: object
    num_physical_experts: object

    @property
    def num_slots(self) -> int:
        """Expert slots the file asks for, 0 when it names none."""
        slots = self.num_physical_experts if self.format == 2 else self.num_experts
        return slots if type(slots) is int and slots > 0 else 0


# The environment's placement file is read once per process and path: every
# MoE layer sizes its slots from the same bytes, and those are the bytes whose
# digest the EP ranks compare at warmup.
_identities: dict[str, PlacementFileIdentity] = {}


def expert_placement_identity() -> PlacementFileIdentity | None:
    """Identity of the placement file named by the environment.

    Returns:
        None when the environment names no file.

    Raises:
        ValueError: The file cannot be read as JSON.
    """
    path = expert_placement_path()
    if path is None:
        return None
    identity = _identities.get(path)
    if identity is None:
        try:
            with open(path, "rb") as handle:
                raw = handle.read()
            data = json.loads(raw)
        except (OSError, ValueError) as error:
            raise _fail(path, f"cannot read the placement ({error})") from error
        fields = data if isinstance(data, dict) else {}
        identity = PlacementFileIdentity(
            path=path,
            digest=_digest(hashlib.sha256(raw).hexdigest()),
            format={PLACEMENT_FORMAT: 1, PLACEMENT_FORMAT_REPLICAS: 2}.get(
                fields.get("format"), 0
            ),
            num_experts=fields.get("num_experts"),
            num_physical_experts=fields.get("num_physical_experts"),
        )
        _identities[path] = identity
    return identity


def load_expert_placement(
    path: str, *, num_layers: int, num_experts: int, ep_size: int
) -> FileExpertPlacement:
    """Read and validate a placement file for one model.

    Args:
        path: The placement file.
        num_layers: MoE layers of the model.
        num_experts: Routed experts per layer.
        ep_size: The model's expert-parallel group size.

    Returns:
        The placement.

    Raises:
        ValueError: The file is unreadable, is not a placement for this model
            and EP size, or a layer's row is not a permutation.
    """
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
        data = json.loads(raw)
    except (OSError, ValueError) as error:
        raise _fail(path, f"cannot read the placement ({error})") from error
    if not isinstance(data, dict) or data.get("format") not in (
        PLACEMENT_FORMAT,
        PLACEMENT_FORMAT_REPLICAS,
    ):
        raise _fail(
            path, f"not a {PLACEMENT_FORMAT} or {PLACEMENT_FORMAT_REPLICAS} file"
        )
    replicas = data["format"] == PLACEMENT_FORMAT_REPLICAS
    if data.get("num_experts") != num_experts or data.get("num_layers") != num_layers:
        raise _fail(
            path,
            f"placement is for {data.get('num_experts')} experts in "
            f"{data.get('num_layers')} layers, the model has {num_experts} in "
            f"{num_layers}",
        )
    ep_sizes = data.get("ep_sizes")
    if not isinstance(ep_sizes, list) or ep_size not in ep_sizes:
        raise _fail(
            path,
            f"placement was solved for EP sizes {ep_sizes}, the model runs EP {ep_size}",
        )
    if ep_size < 2 or num_experts % ep_size:
        raise _fail(path, f"{num_experts} experts do not divide over EP {ep_size}")
    rows = data.get("physical_to_logical")
    if not isinstance(rows, list) or len(rows) != num_layers:
        raise _fail(path, f"physical_to_logical must hold {num_layers} layer rows")
    num_slots = num_experts
    homes: list | None = None
    if replicas:
        num_slots = data.get("num_physical_experts")
        if (
            type(num_slots) is not int
            or num_slots < num_experts
            or num_slots % ep_size
            or num_slots // ep_size > _MAX_SLOTS_PER_RANK
        ):
            raise _fail(
                path,
                f"num_physical_experts={num_slots} must be at least {num_experts}, "
                f"a multiple of EP {ep_size} and at most {_MAX_SLOTS_PER_RANK} per rank",
            )
        homes = data.get("home_slot")
        if not isinstance(homes, list) or len(homes) != num_layers:
            raise _fail(path, f"home_slot must hold {num_layers} layer rows")
    expected = list(range(num_experts))
    physical_to_logical, logical_to_physical = [], []
    for layer_id, row in enumerate(rows):
        if (
            not isinstance(row, list)
            or len(row) != num_slots
            or any(type(expert) is not int for expert in row)
        ):
            raise _fail(path, f"layer {layer_id} does not list {num_slots} expert ids")
        if homes is None:
            if sorted(row) != expected:
                raise _fail(
                    path,
                    f"layer {layer_id} is not a permutation of {num_experts} expert ids",
                )
            home = [0] * num_experts
            for slot, expert in enumerate(row):
                home[expert] = slot
        else:
            home = homes[layer_id]
            if (
                not isinstance(home, list)
                or len(home) != num_experts
                or any(type(slot) is not int for slot in home)
                or any(not 0 <= expert < num_experts for expert in row)
                or any(
                    not 0 <= slot < num_slots or row[slot] != expert
                    for expert, slot in enumerate(home)
                )
            ):
                raise _fail(
                    path,
                    f"layer {layer_id}: every expert needs a home slot that holds "
                    f"it, and every slot one of the {num_experts} experts",
                )
        physical_to_logical.append(tuple(row))
        logical_to_physical.append(tuple(home))
    return FileExpertPlacement(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        ep_size=ep_size,
        num_layers=num_layers,
        num_logical_experts=num_experts,
        num_physical_experts=num_slots,
        physical_to_logical=tuple(physical_to_logical),
        logical_to_physical=tuple(logical_to_physical),
    )


def expert_placement_path() -> str | None:
    """The placement file named by the environment, or None when unset."""
    return os.environ.get(EXPERT_PLACEMENT_ENV) or None


def expert_placement_slots(num_experts: int) -> int:
    """Expert slots an MoE stack of ``num_experts`` routed experts is built with.

    ``num_experts`` itself unless the environment names a placement file for
    exactly that many experts that replicates some of them. The MoE layers
    size their weights from this before any weight is loaded; the model
    validates the whole file when it loads them.

    Args:
        num_experts: Routed (logical) experts per layer of the stack.

    Returns:
        The file's ``num_physical_experts``, or ``num_experts``.

    Raises:
        ValueError: The file named by the environment cannot be read.
    """
    identity = expert_placement_identity()
    if identity is None or identity.format != 2 or identity.num_experts != num_experts:
        return num_experts
    slots = identity.num_physical_experts
    if type(slots) is not int or slots < num_experts:
        raise _fail(identity.path, f"num_physical_experts={slots} is not a slot count")
    return slots


class LayerLoadCounter:
    """One MoE layer's route counters, on the GPU.

    Args:
        profile: The process-wide profile this layer reports into.
        group: Key of the MoE stack the layer belongs to.
        layer_index: The layer.
        num_experts: Routed experts of the layer.
        top_k: Routes per row.
    """

    def __init__(
        self,
        profile: ExpertLoadProfile,
        group: str,
        layer_index: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        self.profile = profile
        self.group = group
        self.layer_index = layer_index
        self.num_experts = num_experts
        self.top_k = top_k
        # [len(LOAD_MODES), num_experts] int64, allocated by ``allocate``.
        self.routes: torch.Tensor | None = None
        # Python executions of the layer per mode: eager forwards (a rank
        # that holds no row in a collective launch included) and graph
        # captures. A graph replay adds routes but runs no Python.
        self.forwards = [0] * len(LOAD_MODES)
        # Of ``forwards``, the graph captures. Each stands for any number of
        # replays, which add routes but are counted nowhere.
        self.captures = [0] * len(LOAD_MODES)
        # Launches skipped because the counters were not allocated yet.
        self.dropped = 0

    def allocate(self, device: torch.device, max_rows: int | None = None) -> None:
        """Create the counters on ``device``; call outside CUDA graph capture.

        Args:
            device: The layer's device.
            max_rows: Most rows one forward of this layer can hold on this
                rank, when the layer knows it. The profile's buffer of ones
                is then sized for it now, so no forward has to grow it.
        """
        if self.routes is None:
            self.routes = torch.zeros(
                len(LOAD_MODES), self.num_experts, dtype=torch.int64, device=device
            )
        if max_rows is not None and max_rows > 0:
            self.profile.reserve_ones(max_rows * self.top_k, device)

    def record(self, topk_ids: torch.Tensor, mode: int) -> None:
        """Count this forward's routes: one index-add launch, no host sync.

        Args:
            topk_ids: ``[rows, top_k]`` logical expert ids (int32 or int64).
            mode: Index into ``LOAD_MODES``.
        """
        count = topk_ids.numel()
        if count == 0:
            # An idle rank still takes part in the collective launch: count
            # the launch, launch nothing.
            self.forwards[mode] += 1
            return
        capturing = torch.cuda.is_current_stream_capturing()
        if self.routes is None:
            if capturing:
                # A fill captured into a graph would clear the counters on
                # every replay.
                self.dropped += 1
                return
            self.allocate(topk_ids.device)
        ones = self.profile.ones(count, topk_ids.device, capturing)
        if ones is None:
            self.dropped += 1
            return
        if topk_ids.dtype not in (torch.int32, torch.int64):
            topk_ids = topk_ids.to(torch.int64)
        self.routes[mode].index_add_(0, topk_ids.reshape(-1), ones)
        self.forwards[mode] += 1
        if capturing:
            self.captures[mode] += 1
        self.profile.forward_done(self, capturing)


class ExpertLoadProfile:
    """Per-layer route counters of one rank, written to ``rank<r>.json``.

    Args:
        directory: Where the profile is written.
        interval: Forwards between two writes.
    """

    def __init__(self, directory: str, interval: int) -> None:
        self.directory = directory
        self.interval = interval
        self.counters: list[LayerLoadCounter] = []
        self.groups: dict[str, dict[str, int]] = {}
        self.ep_rank: int | None = None
        self.ep_size: int | None = None
        self._ones: torch.Tensor | None = None
        # Buffers ``_ones`` replaced. A captured graph's index-add reads its
        # buffer by address on every replay, so none is ever freed.
        self._superseded_ones: list[torch.Tensor] = []
        self._forwards_since_write = 0
        self._last_write = time.monotonic()
        self._writes = 0
        os.makedirs(directory, exist_ok=True)
        atexit.register(self._write_at_exit)

    def counter(
        self,
        *,
        layer_index: int,
        num_experts: int,
        top_k: int,
        ep_rank: int,
        ep_size: int,
    ) -> LayerLoadCounter:
        """Register one MoE layer and return its counter."""
        if self.ep_rank is None:
            self.ep_rank, self.ep_size = ep_rank, ep_size
        elif (self.ep_rank, self.ep_size) != (ep_rank, ep_size):
            raise ValueError(
                f"{LOAD_PROFILE_ENV}: one process holds EP ranks "
                f"{(self.ep_rank, self.ep_size)} and {(ep_rank, ep_size)}"
            )
        # The target's layers and a draft model's layers share layer indices
        # but not their expert count.
        group = f"e{num_experts}k{top_k}"
        self.groups.setdefault(group, {"num_experts": num_experts, "top_k": top_k})
        counter = LayerLoadCounter(self, group, layer_index, num_experts, top_k)
        self.counters.append(counter)
        return counter

    def ones(
        self, count: int, device: torch.device, capturing: bool
    ) -> torch.Tensor | None:
        """``count`` int64 ones on ``device``; None when it cannot be made now.

        Under capture the buffer is never grown: an allocation captured into
        a graph would not be the one a replay reads.
        """
        if not self._has_ones(count, device):
            if capturing:
                return None
            self._grow_ones(count, device)
        return self._ones[:count]

    def reserve_ones(self, count: int, device: torch.device) -> None:
        """Size the buffer for ``count`` routes; call outside graph capture."""
        if not self._has_ones(count, device):
            self._grow_ones(count, device)

    def _has_ones(self, count: int, device: torch.device) -> bool:
        return (
            self._ones is not None
            and self._ones.numel() >= count
            and self._ones.device == device
        )

    def _grow_ones(self, count: int, device: torch.device) -> None:
        if self._ones is not None:
            # Graphs captured so far keep reading the old buffer.
            self._superseded_ones.append(self._ones)
        self._ones = torch.ones(
            max(1 << 16, 1 << (count - 1).bit_length()),
            dtype=torch.int64,
            device=device,
        )

    def forward_done(self, counter: LayerLoadCounter, capturing: bool) -> None:
        """Called after every recorded launch; the first layer paces the writes."""
        if counter is not self.counters[0]:
            return
        self._forwards_since_write += 1
        due = self._forwards_since_write >= self.interval or (
            time.monotonic() - self._last_write >= _PROFILE_MAX_AGE_S
        )
        if due and not capturing:
            self.write("eager_forward")

    def snapshot(self, trigger: str = "explicit") -> dict:
        """The counters as plain data; copies them to the host.

        Args:
            trigger: What asked for the snapshot: ``eager_forward`` (the
                periodic write), ``exit`` or ``explicit``.
        """
        groups: dict[str, dict] = {}
        replayed: set[str] = set()
        for key, info in self.groups.items():
            members = sorted(
                (c for c in self.counters if c.group == key),
                key=lambda c: c.layer_index,
            )
            layers = []
            for counter in members:
                routes = (
                    counter.routes.cpu().tolist()
                    if counter.routes is not None
                    else [[0] * counter.num_experts for _ in LOAD_MODES]
                )
                modes = {}
                for index, mode in enumerate(LOAD_MODES):
                    total = sum(routes[index])
                    if total == 0 and counter.forwards[index] == 0:
                        continue
                    modes[mode] = {
                        "forwards": counter.forwards[index],
                        "graph_captures": counter.captures[index],
                        # Exact up to this write, the replays of captured
                        # graphs included: every row has top_k routes.
                        "rows": total // counter.top_k,
                        "routes": routes[index],
                    }
                    if counter.captures[index] > 0:
                        replayed.add(mode)
                layers.append(
                    {
                        "layer": counter.layer_index,
                        "dropped": counter.dropped,
                        "modes": modes,
                    }
                )
            groups[key] = {**info, "num_layers": len(members), "layers": layers}
        none = [0] * len(LOAD_MODES)
        forwards = self.counters[0].forwards if self.counters else none
        captures = self.counters[0].captures if self.counters else none
        return {
            "format": PROFILE_FORMAT,
            "rank": self.ep_rank,
            "ep_size": self.ep_size,
            "pid": os.getpid(),
            "writes": self._writes + 1,
            "written_unix": time.time(),
            "note": (
                "routes[e] counts this rank's own rows routed to logical expert e; "
                "forwards counts Python executions (eager forwards, also with no "
                "row on this rank, and graph captures), rows is routes / top_k"
            ),
            # What the counts stand for.
            "coverage": {
                "trigger": trigger,
                # Python executions of the first registered layer, the one
                # that paces the writes, per forward kind.
                "eager_forwards": {
                    mode: forwards[index] - captures[index]
                    for index, mode in enumerate(LOAD_MODES)
                },
                "graph_captures": dict(zip(LOAD_MODES, captures)),
                # Kinds whose routes include graph replays in some layer:
                # those launches run no Python, are not in "forwards", and
                # trigger no write.
                "modes_with_graph_replays": [
                    mode for mode in LOAD_MODES if mode in replayed
                ],
                "written": (
                    "only from an eager forward (every "
                    f"{self.interval} forwards of the first layer, or "
                    f"{_PROFILE_MAX_AGE_S:.0f} s after the last write) and at "
                    "interpreter exit, which a killed rank process never reaches"
                ),
                "warning": (
                    "A mode listed in modes_with_graph_replays holds the replays "
                    "up to this write only. If the process serves that mode from "
                    "graphs alone, this write is the last eager forward, usually "
                    "the start-up warm-up on dummy inputs: do not solve a "
                    "placement for that mode from this file."
                ),
            },
            "groups": groups,
        }

    def write(self, trigger: str = "explicit") -> None:
        """Write ``rank<r>.json`` atomically.

        Args:
            trigger: Recorded in the file's ``coverage``; see ``snapshot``.
        """
        if self.ep_rank is None:
            return
        data = self.snapshot(trigger)
        path = os.path.join(self.directory, f"rank{self.ep_rank}.json")
        temporary = f"{path}.tmp.{os.getpid()}"
        with open(temporary, "w") as handle:
            json.dump(data, handle)
        os.replace(temporary, path)
        self._writes += 1
        self._forwards_since_write = 0
        self._last_write = time.monotonic()

    def _write_at_exit(self) -> None:
        try:
            self.write("exit")
        except Exception as error:  # the interpreter is shutting down
            logger.warning(f"{LOAD_PROFILE_ENV}: final write failed: {error}")


_profile: ExpertLoadProfile | None = None


def expert_load_profile() -> ExpertLoadProfile | None:
    """The process-wide load profile, or None when the environment sets none."""
    global _profile
    directory = os.environ.get(LOAD_PROFILE_ENV) or None
    if directory is None:
        return None
    if _profile is None:
        interval = int(
            os.environ.get(LOAD_PROFILE_INTERVAL_ENV) or _DEFAULT_PROFILE_INTERVAL
        )
        if interval < 1:
            raise ValueError(f"{LOAD_PROFILE_INTERVAL_ENV} must be positive")
        _profile = ExpertLoadProfile(directory, interval)
        logger.info(
            f"{LOAD_PROFILE_ENV}: recording MoE expert load into {directory} "
            f"every {interval} forwards"
        )
    return _profile
