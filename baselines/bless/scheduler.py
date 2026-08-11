#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class KernelProfile:
    name: str
    cumulative_us: Mapping[int, float]
    duration_us: Mapping[int, float]
    native_share: int

    def __post_init__(self) -> None:
        shares = set(self.cumulative_us)
        if not self.name or shares != set(self.duration_us):
            raise ValueError("kernel profile shares must match")
        if self.native_share not in shares:
            raise ValueError("native share must be profiled")
        for share in shares:
            if not 0 < share <= 100:
                raise ValueError("profile shares must be in (0, 100]")
            values = (self.cumulative_us[share], self.duration_us[share])
            if any(not isfinite(value) or value <= 0 for value in values):
                raise ValueError("profile times must be finite and positive")


@dataclass(frozen=True)
class RequestState:
    request_id: str
    target_us: float
    elapsed_us: float
    kernels: Sequence[KernelProfile]
    next_kernel: int = 0

    def __post_init__(self) -> None:
        if not self.request_id or not self.kernels:
            raise ValueError("request id and kernels are required")
        if not isfinite(self.target_us) or self.target_us <= 0:
            raise ValueError("target must be finite and positive")
        if not isfinite(self.elapsed_us) or self.elapsed_us < 0:
            raise ValueError("elapsed time must be finite and nonnegative")
        if not 0 <= self.next_kernel < len(self.kernels):
            raise ValueError("next kernel is outside the request")

    def relative_progress(self) -> float:
        expected = self.kernels[self.next_kernel].cumulative_us[100]
        return self.elapsed_us / expected


@dataclass(frozen=True)
class ScheduledKernel:
    request_id: str
    kernel_index: int
    profile: KernelProfile


@dataclass(frozen=True)
class SquadConfiguration:
    shares: Mapping[str, int] | None
    predicted_us: float
    estimator: str


def form_kernel_squad(
    requests: Sequence[RequestState], maximum_kernels: int = 6
) -> list[ScheduledKernel]:
    """Implement BLESS Section 4.3.2's relative-progress squad rule."""
    if maximum_kernels <= 0:
        raise ValueError("maximum_kernels must be positive")
    if not requests:
        return []

    cursors = {request.request_id: request.next_kernel for request in requests}
    if len(cursors) != len(requests):
        raise ValueError("request ids must be unique")
    squad: list[ScheduledKernel] = []

    while len(squad) < maximum_kernels:
        active = [
            request
            for request in requests
            if cursors[request.request_id] < len(request.kernels)
        ]
        if not active:
            break

        def urgency(request: RequestState) -> tuple[float, int]:
            cursor = cursors[request.request_id]
            expected = request.kernels[cursor].cumulative_us[100]
            return request.elapsed_us / expected, requests.index(request)

        selected = min(active, key=urgency)
        cursor = cursors[selected.request_id]
        squad.append(
            ScheduledKernel(selected.request_id, cursor, selected.kernels[cursor])
        )
        cursors[selected.request_id] = cursor + 1
        if cursor + 1 == len(selected.kernels):
            break

    return squad


def strict_share_configurations(
    request_ids: Sequence[str], allowed_shares: Iterable[int]
) -> list[dict[str, int]]:
    """Enumerate strict spatial partitions whose shares sum to 100%."""
    if not request_ids or len(set(request_ids)) != len(request_ids):
        raise ValueError("request ids must be nonempty and unique")
    shares = sorted(set(allowed_shares))
    if any(share <= 0 or share >= 100 for share in shares):
        raise ValueError("strict shares must be in (0, 100)")
    return [
        dict(zip(request_ids, candidate, strict=True))
        for candidate in product(shares, repeat=len(request_ids))
        if sum(candidate) == 100
    ]


def interference_free_duration_us(
    squad: Sequence[ScheduledKernel], shares: Mapping[str, int]
) -> float:
    """Equation 1: maximum per-request stack under strict isolation."""
    stacks: dict[str, float] = {}
    for item in squad:
        if item.request_id not in shares:
            raise ValueError("configuration omits an active request")
        share = shares[item.request_id]
        if share not in item.profile.duration_us:
            raise ValueError("configuration uses an unprofiled share")
        stacks[item.request_id] = (
            stacks.get(item.request_id, 0.0) + item.profile.duration_us[share]
        )
    return max(stacks.values(), default=0.0)


def workload_equivalence_duration_us(
    squad: Sequence[ScheduledKernel], request_order: Sequence[str]
) -> float:
    """Equation 2's breadth-first unrestricted-kernel estimate."""
    queues = {
        request_id: [item for item in squad if item.request_id == request_id]
        for request_id in request_order
    }
    if set(item.request_id for item in squad) - set(queues):
        raise ValueError("request order omits an active request")
    duration = 0.0
    for depth in range(max((len(queue) for queue in queues.values()), default=0)):
        for request_id in request_order:
            queue = queues[request_id]
            if depth < len(queue):
                profile = queue[depth].profile
                duration += profile.duration_us[profile.native_share]
    return duration


def choose_configuration(
    squad: Sequence[ScheduledKernel], allowed_shares: Iterable[int]
) -> SquadConfiguration:
    if not squad:
        raise ValueError("cannot configure an empty squad")
    request_ids = list(dict.fromkeys(item.request_id for item in squad))
    candidates = [
        SquadConfiguration(
            shares=shares,
            predicted_us=interference_free_duration_us(squad, shares),
            estimator="interference-free",
        )
        for shares in strict_share_configurations(request_ids, allowed_shares)
    ]
    candidates.append(
        SquadConfiguration(
            shares=None,
            predicted_us=workload_equivalence_duration_us(squad, request_ids),
            estimator="workload-equivalence",
        )
    )
    return min(
        candidates,
        key=lambda candidate: (
            candidate.predicted_us,
            candidate.estimator != "interference-free",
            tuple((candidate.shares or {}).values()),
        ),
    )
