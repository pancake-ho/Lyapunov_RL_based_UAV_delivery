from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from env.action_types import SlowAction
from env.validators import (
    parse_slow_action,
    validate_slow_action_strict,
)


SlowActionDict = Dict[str, np.ndarray]
ScoreActionsFn = Callable[
    [Sequence[SlowActionDict]],
    Sequence[float],
]


@dataclass(frozen=True)
class PairEdge:
    """
    One provider-user edge used by the additive matching surrogate.

    weight:
        Positive value means that the pair is predicted to reduce the
        frame/round cumulative DPP cost relative to its outside option.

    pair_cost:
        Full-frame score returned by the Fast-policy rollout for the
        corresponding one-pair candidate action. For a UAV edge this
        value includes the fixed UAV hiring cost; ``weight`` removes that
        fixed cost so it can be charged once per hired UAV after matching.
    """

    provider: int
    user: int
    weight: float
    pair_cost: float


@dataclass(frozen=True)
class MatchingSelection:
    action: SlowActionDict
    predicted_round_cost: float

    # Exact full-round stage scores.
    baseline_cost: float
    rsu_only_cost: float
    provisional_final_cost: float
    chosen_stage: str

    # RSU additive-surrogate diagnostics.
    rsu_candidate_edges: int
    rsu_positive_candidate_edges: int
    best_rsu_edge_weight: float
    rsu_matches: Tuple[PairEdge, ...]
    rsu_weight_sum: float

    # UAV candidate-edge diagnostics.
    uav_candidate_edges: int
    uav_positive_candidate_edges: int
    best_uav_edge_weight: float

    # UAV b-matching result before the fixed hiring-cost gate.
    provisional_uav_match_count: int
    provisional_uav_provider_count: int
    provisional_uav_service_weight_sum: float
    provisional_uav_hiring_cost_sum: float
    provisional_uav_net_weight_sum: float
    best_uav_provider_net_gain: float

    # UAV matching retained after the fixed hiring-cost gate.
    uav_matches: Tuple[PairEdge, ...]
    uav_service_weight_sum: float
    uav_net_weight_sum: float
    hired_uavs: Tuple[int, ...]

def _copy_action(
    action: Mapping[str, np.ndarray],
) -> SlowActionDict:
    return {
        key: np.asarray(value, dtype=np.int32).copy()
        for key, value in action.items()
    }


def _zero_action(env) -> SlowActionDict:
    return {
        "rsu_scheduling": np.zeros(
            (int(env.num_rsu), int(env.num_user)),
            dtype=np.int32,
        ),
        "uav_hiring": np.zeros(
            int(env.num_uav),
            dtype=np.int32,
        ),
        "uav_scheduling": np.zeros(
            (int(env.num_uav), int(env.num_user)),
            dtype=np.int32,
        ),
    }


def _validate(
    env,
    action: Mapping[str, np.ndarray],
    *,
    forbid_empty_hiring: bool,
) -> SlowAction:
    parsed = parse_slow_action(
        dict(action),
        env.cfg,
    )
    validate_slow_action_strict(
        parsed,
        env.cfg,
        user_region=np.asarray(
            env.user_region,
            dtype=np.int32,
        ),
        requested_content=np.asarray(
            env.requested_content,
            dtype=np.int32,
        ),
        uav_cached_content=np.asarray(
            env.uav_cached_content,
            dtype=np.int32,
        ),
        forbid_empty_hiring=bool(
            forbid_empty_hiring
        ),
    )
    return parsed


def _score_many(
    evaluate: ScoreActionsFn,
    actions: Sequence[SlowActionDict],
) -> np.ndarray:
    if not actions:
        return np.zeros(0, dtype=np.float64)

    scores = np.asarray(
        list(evaluate(actions)),
        dtype=np.float64,
    )
    if scores.shape != (len(actions),):
        raise RuntimeError(
            "Slow matching evaluator returned an invalid score shape: "
            f"expected={(len(actions),)}, got={scores.shape}."
        )
    return scores


def _score_one(
    evaluate: ScoreActionsFn,
    action: SlowActionDict,
) -> float:
    return float(
        _score_many(
            evaluate,
            [action],
        )[0]
    )


def _hiring_cost_vector(env) -> np.ndarray:
    """
    Return W_hire * D_u^hire for every UAV.

    SlowDPPEvaluator already adds the same fixed cost once for every
    ``uav_hiring[u] == 1``. We use this vector only to remove that fixed
    cost from one-UAV-edge rollout scores and then apply it once after the
    UAV b-matching.
    """
    raw = np.asarray(
        env._hire_cost(),  # existing Env API
        dtype=np.float64,
    ).reshape(-1)

    if raw.shape != (int(env.num_uav),):
        raise ValueError(
            "UAV hiring-cost shape mismatch: "
            f"expected={(int(env.num_uav),)}, got={raw.shape}."
        )

    weight = float(
        getattr(
            env.cfg,
            "hire_weight",
            1.0,
        )
    )
    return weight * raw


def solve_max_weight_b_matching(
    *,
    edges: Sequence[PairEdge],
    provider_count: int,
    user_count: int,
    capacities: Sequence[int],
    min_weight: float,
) -> Tuple[PairEdge, ...]:
    """
    Solve a maximum-weight bipartite b-matching by slot expansion.

    Each provider p with capacity b_p is replicated into b_p unit-capacity
    provider slots. Users retain unit capacity. One private dummy column
    per provider slot provides the unmatched option with weight zero.

    The resulting assignment is solved exactly by Hungarian
    ``linear_sum_assignment``. Therefore this function is exact for the
    additive edge-weight surrogate. It is NOT a proof of global optimality
    for the original non-additive full-round DPP objective.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Slow DPP maximum-weight matching requires scipy. "
            "Install/activate the existing lab environment that provides scipy."
        ) from exc

    provider_count = int(provider_count)
    user_count = int(user_count)

    if provider_count < 0 or user_count < 0:
        raise ValueError(
            "provider_count and user_count must be nonnegative."
        )

    caps = np.asarray(
        capacities,
        dtype=np.int64,
    )
    if caps.shape != (provider_count,):
        raise ValueError(
            "capacities shape mismatch: "
            f"expected={(provider_count,)}, got={caps.shape}."
        )
    if np.any(caps < 0):
        raise ValueError(
            "Provider capacities must be nonnegative."
        )

    slot_provider = []
    for provider, capacity in enumerate(caps):
        slot_provider.extend(
            [int(provider)] * int(capacity)
        )

    slot_count = len(slot_provider)
    if slot_count == 0 or user_count == 0:
        return tuple()

    edge_lookup: Dict[
        tuple[int, int],
        PairEdge,
    ] = {}
    for edge in edges:
        provider = int(edge.provider)
        user = int(edge.user)
        if not (
            0 <= provider < provider_count
        ):
            raise ValueError(
                f"Invalid provider index: {provider}."
            )
        if not (0 <= user < user_count):
            raise ValueError(
                f"Invalid user index: {user}."
            )
        key = (provider, user)
        if key in edge_lookup:
            raise ValueError(
                "Duplicate provider-user matching edge: "
                f"{key}."
            )
        edge_lookup[key] = edge

    invalid_weight = -1.0e30
    weight_matrix = np.full(
        (
            slot_count,
            user_count + slot_count,
        ),
        invalid_weight,
        dtype=np.float64,
    )
    # Private dummy/unmatched columns, all zero.
    weight_matrix[
        :,
        user_count:
    ] = 0.0

    threshold = float(min_weight)

    for row_idx, provider in enumerate(
        slot_provider
    ):
        for user in range(user_count):
            edge = edge_lookup.get(
                (int(provider), int(user))
            )
            if edge is None:
                continue

            weight = float(edge.weight)
            if (
                math.isfinite(weight)
                and weight > threshold
            ):
                weight_matrix[
                    row_idx,
                    user,
                ] = weight

    row_ind, col_ind = (
        linear_sum_assignment(
            -weight_matrix
        )
    )

    selected = []
    used_users: set[int] = set()

    for row_idx, col_idx in zip(
        row_ind,
        col_ind,
    ):
        user = int(col_idx)
        if user >= user_count:
            continue

        provider = int(
            slot_provider[int(row_idx)]
        )
        edge = edge_lookup.get(
            (provider, user)
        )
        if edge is None:
            raise RuntimeError(
                "Hungarian solver selected an invalid edge."
            )

        if not (
            math.isfinite(float(edge.weight))
            and float(edge.weight) > threshold
        ):
            raise RuntimeError(
                "Hungarian solver selected a non-beneficial edge."
            )

        if user in used_users:
            raise RuntimeError(
                "Matching assigned one user more than once."
            )
        used_users.add(user)
        selected.append(edge)

    selected.sort(
        key=lambda edge: (
            int(edge.provider),
            int(edge.user),
        )
    )
    return tuple(selected)


def _action_tie_key(
    action: Mapping[str, np.ndarray],
) -> tuple:
    """
    Deterministic structural priority for near-equal full-round costs.

    1) fewer hired UAVs,
    2) more RSU links,
    3) fewer UAV links,
    4) deterministic binary action bytes.
    """
    rsu = np.asarray(
        action["rsu_scheduling"],
        dtype=np.int32,
    )
    hiring = np.asarray(
        action["uav_hiring"],
        dtype=np.int32,
    )
    uav = np.asarray(
        action["uav_scheduling"],
        dtype=np.int32,
    )

    action_bytes = b"|".join(
        np.asarray(
            action[name],
            dtype=np.int8,
        ).tobytes()
        for name in (
            "rsu_scheduling",
            "uav_hiring",
            "uav_scheduling",
        )
    )

    return (
        int(hiring.sum()),
        -int(rsu.sum()),
        int(uav.sum()),
        action_bytes,
    )


def _choose_exact_stage(
    *,
    candidates: Sequence[
        tuple[
            str,
            SlowActionDict,
            float,
        ]
    ],
    tolerance: float,
) -> tuple[
    str,
    SlowActionDict,
    float,
]:
    finite = [
        (
            str(name),
            _copy_action(action),
            float(score),
        )
        for name, action, score in candidates
        if math.isfinite(float(score))
    ]

    if not finite:
        raise RuntimeError(
            "All Slow matching stage candidates are infeasible."
        )

    best_name, best_action, best_score = (
        finite[0]
    )
    tol = max(
        float(tolerance),
        0.0,
    )

    for name, action, score in finite[1:]:
        if score < best_score - tol:
            (
                best_name,
                best_action,
                best_score,
            ) = (
                name,
                action,
                score,
            )
            continue

        if (
            abs(score - best_score) <= tol
            and _action_tie_key(action)
            < _action_tie_key(best_action)
        ):
            (
                best_name,
                best_action,
                best_score,
            ) = (
                name,
                action,
                score,
            )

    return (
        best_name,
        best_action,
        float(best_score),
    )


def select_dpp_max_weight_matching(
    *,
    env,
    evaluate: ScoreActionsFn,
    min_edge_weight: float,
    forbid_empty_hiring: bool = True,
) -> MatchingSelection:
    """
    Select the round-fixed Slow action using DPP-based maximum-weight
    b-matching.

    Latest Track-A decision implemented here
    -----------------------------------------
    1) Evaluate the all-unmatched outside option with the Fast policy over
       one complete round/frame.
    2) For every feasible RSU-user pair, evaluate its predicted cumulative
       Fast DPP cost and construct

           w^R_mn = C_outside - C^R_mn.

       Solve maximum-weight b-matching with RSU capacity.
    3) Freeze that RSU matching. Only RSU-unmatched users are residual
       users. For every cache-compatible residual UAV-user pair, evaluate
       one complete frame and remove the one-time UAV hiring cost:

           C^U,fast_un = C^U,total_un - H_u
           w^U_un      = C_R - C^U,fast_un.

       Solve maximum-weight b-matching with UAV capacity. For every UAV,
       retain its selected edges only when

           sum_n w^U_un - H_u > threshold.

    4) Re-evaluate the complete matched actions. Because pair weights form
       an additive surrogate while the original Fast-policy rollout is
       coupled through queue/battery/policy dynamics, compare the exact
       full-round scores of:
           outside option,
           RSU-only MWM,
           RSU+UAV MWM,
       and keep the lowest-cost feasible complete action.

    Thus the Hungarian result is exact for the constructed additive
    b-matching surrogate, while global optimality for the original coupled
    round DPP objective is intentionally NOT claimed.
    """
    if int(env.round_slot) != 0:
        raise RuntimeError(
            "Slow matching must be selected only at a round boundary: "
            f"round_slot={env.round_slot}."
        )

    if int(env.num_rsu) != int(env.num_uav):
        raise ValueError(
            "Current one-UAV-per-region mapping requires "
            "num_rsu == num_uav."
        )

    threshold = max(
        float(min_edge_weight),
        0.0,
    )

    region = np.asarray(
        env.user_region,
        dtype=np.int32,
    ).reshape(-1)
    requested = np.asarray(
        env.requested_content,
        dtype=np.int32,
    ).reshape(-1)
    cached = np.asarray(
        env.uav_cached_content,
        dtype=np.int32,
    ).reshape(-1)

    num_rsu = int(env.num_rsu)
    num_uav = int(env.num_uav)
    num_user = int(env.num_user)

    if region.shape != (num_user,):
        raise ValueError(
            "user_region shape mismatch."
        )
    if requested.shape != (num_user,):
        raise ValueError(
            "requested_content shape mismatch."
        )
    if cached.shape != (num_uav,):
        raise ValueError(
            "uav_cached_content shape mismatch."
        )

    # 1) Outside option.
    baseline_action = _zero_action(env)
    _validate(
        env,
        baseline_action,
        forbid_empty_hiring=forbid_empty_hiring,
    )
    baseline_cost = _score_one(
        evaluate,
        baseline_action,
    )
    if not math.isfinite(
        baseline_cost
    ):
        raise RuntimeError(
            "The all-unmatched Slow outside option produced "
            "a non-finite predicted DPP cost."
        )

    # 2) RSU-user pair costs -> MWM b-matching.
    rsu_pair_actions = []
    rsu_pair_keys = []

    for user in range(num_user):
        provider = int(region[user])
        candidate = _zero_action(env)
        candidate[
            "rsu_scheduling"
        ][provider, user] = 1

        _validate(
            env,
            candidate,
            forbid_empty_hiring=forbid_empty_hiring,
        )
        rsu_pair_actions.append(
            candidate
        )
        rsu_pair_keys.append(
            (provider, user)
        )

    rsu_pair_scores = _score_many(
        evaluate,
        rsu_pair_actions,
    )

    rsu_edges = []
    for (
        (provider, user),
        pair_cost,
    ) in zip(
        rsu_pair_keys,
        rsu_pair_scores,
    ):
        weight = (
            baseline_cost
            - float(pair_cost)
            if math.isfinite(
                float(pair_cost)
            )
            else -math.inf
        )
        rsu_edges.append(
            PairEdge(
                provider=int(provider),
                user=int(user),
                weight=float(weight),
                pair_cost=float(
                    pair_cost
                ),
            )
        )
    
    rsu_finite_weights = [
        float(edge.weight)
        for edge in rsu_edges
        if math.isfinite(
            float(edge.weight)
        )
    ]

    rsu_positive_candidate_edges = int(
        sum(
            1
            for weight in rsu_finite_weights
            if weight > threshold
        )
    )

    best_rsu_edge_weight = (
        float(
            max(rsu_finite_weights)
        )
        if rsu_finite_weights
        else float("nan")
    )    

    rsu_matches = (
        solve_max_weight_b_matching(
            edges=rsu_edges,
            provider_count=num_rsu,
            user_count=num_user,
            capacities=[
                int(env.cfg.rsu_capacity)
                for _ in range(num_rsu)
            ],
            min_weight=threshold,
        )
    )

    rsu_action = _zero_action(env)
    for edge in rsu_matches:
        rsu_action[
            "rsu_scheduling"
        ][
            int(edge.provider),
            int(edge.user),
        ] = 1

    _validate(
        env,
        rsu_action,
        forbid_empty_hiring=forbid_empty_hiring,
    )
    rsu_only_cost = _score_one(
        evaluate,
        rsu_action,
    )

    if not math.isfinite(
        rsu_only_cost
    ):
        rsu_action = _copy_action(
            baseline_action
        )
        rsu_only_cost = float(
            baseline_cost
        )
        rsu_matches = tuple()

    # 3) Residual UAV-user pair costs -> UAV MWM + fixed hiring gate.
    rsu_scheduled = np.asarray(
        rsu_action[
            "rsu_scheduling"
        ],
        dtype=np.int32,
    ).sum(axis=0) > 0

    hiring_costs = (
        _hiring_cost_vector(env)
    )

    uav_pair_actions = []
    uav_pair_keys = []

    for user in range(num_user):
        if bool(rsu_scheduled[user]):
            continue

        provider = int(region[user])
        if (
            int(requested[user])
            != int(cached[provider])
        ):
            continue

        candidate = _copy_action(
            rsu_action
        )
        candidate[
            "uav_hiring"
        ][provider] = 1
        candidate[
            "uav_scheduling"
        ][provider, user] = 1

        _validate(
            env,
            candidate,
            forbid_empty_hiring=forbid_empty_hiring,
        )
        uav_pair_actions.append(
            candidate
        )
        uav_pair_keys.append(
            (provider, user)
        )

    uav_pair_scores = _score_many(
        evaluate,
        uav_pair_actions,
    )

    uav_edges = []
    for (
        (provider, user),
        total_pair_cost,
    ) in zip(
        uav_pair_keys,
        uav_pair_scores,
    ):
        if math.isfinite(
            float(total_pair_cost)
        ):
            # Evaluator score already contains the one-time hiring cost.
            # Remove it for a service-only edge weight; the fixed cost is
            # then charged once after matching.
            fast_only_pair_cost = (
                float(total_pair_cost)
                - float(
                    hiring_costs[
                        provider
                    ]
                )
            )
            weight = (
                float(rsu_only_cost)
                - fast_only_pair_cost
            )
        else:
            weight = -math.inf

        uav_edges.append(
            PairEdge(
                provider=int(provider),
                user=int(user),
                weight=float(weight),
                pair_cost=float(
                    total_pair_cost
                ),
            )
        )

    uav_finite_weights = [
        float(edge.weight)
        for edge in uav_edges
        if math.isfinite(
            float(edge.weight)
        )
    ]

    uav_positive_candidate_edges = int(
        sum(
            1
            for weight in uav_finite_weights
            if weight > threshold
        )
    )

    best_uav_edge_weight = (
        float(
            max(uav_finite_weights)
        )
        if uav_finite_weights
        else float("nan")
    )
    provisional_uav_matches = (
        solve_max_weight_b_matching(
            edges=uav_edges,
            provider_count=num_uav,
            user_count=num_user,
            capacities=[
                int(env.cfg.uav_user_cap)
                for _ in range(num_uav)
            ],
            min_weight=threshold,
        )
    )
    provisional_uav_providers = tuple(
        sorted(
            {
                int(edge.provider)
                for edge
                in provisional_uav_matches
            }
        )
    )

    provisional_uav_service_weight_sum = float(
        sum(
            float(edge.weight)
            for edge
            in provisional_uav_matches
        )
    )

    provisional_uav_hiring_cost_sum = float(
        sum(
            float(
                hiring_costs[
                    provider
                ]
            )
            for provider
            in provisional_uav_providers
        )
    )

    provisional_uav_net_weight_sum = float(
        provisional_uav_service_weight_sum
        - provisional_uav_hiring_cost_sum
    )

    best_uav_provider_net_gain = float(
        "nan"
    )

    by_uav: Dict[
        int,
        list[PairEdge],
    ] = {
        provider: []
        for provider in range(num_uav)
    }
    for edge in provisional_uav_matches:
        by_uav[
            int(edge.provider)
        ].append(edge)

    kept_uav_matches = []
    hired_uavs = []

    for provider in range(num_uav):
        provider_edges = by_uav[
            provider
        ]

        if not provider_edges:
            continue

        service_gain = float(
            sum(
                float(edge.weight)
                for edge
                in provider_edges
            )
        )

        net_gain = float(
            service_gain
            - float(
                hiring_costs[
                    provider
                ]
            )
        )

        if (
            not math.isfinite(
                best_uav_provider_net_gain
            )
            or net_gain
            > best_uav_provider_net_gain
        ):
            best_uav_provider_net_gain = (
                net_gain
            )

        if net_gain > threshold:
            hired_uavs.append(
                int(provider)
            )

            kept_uav_matches.extend(
                provider_edges
            )   

    kept_uav_matches.sort(
        key=lambda edge: (
            int(edge.provider),
            int(edge.user),
        )
    )
    uav_matches = tuple(
        kept_uav_matches
    )

    final_action = _copy_action(
        rsu_action
    )
    for edge in uav_matches:
        provider = int(edge.provider)
        user = int(edge.user)
        final_action[
            "uav_hiring"
        ][provider] = 1
        final_action[
            "uav_scheduling"
        ][provider, user] = 1

    _validate(
        env,
        final_action,
        forbid_empty_hiring=forbid_empty_hiring,
    )
    provisional_final_cost = (
        _score_one(
            evaluate,
            final_action,
        )
    )

    # 4) Exact complete-action safeguard.
    (
        chosen_stage,
        chosen_action,
        chosen_cost,
    ) = _choose_exact_stage(
        candidates=(
            (
                "outside",
                baseline_action,
                baseline_cost,
            ),
            (
                "rsu_matching",
                rsu_action,
                rsu_only_cost,
            ),
            (
                "rsu_uav_matching",
                final_action,
                provisional_final_cost,
            ),
        ),
        tolerance=threshold,
    )

    _validate(
        env,
        chosen_action,
        forbid_empty_hiring=forbid_empty_hiring,
    )

    rsu_weight_sum = float(
        sum(
            float(edge.weight)
            for edge in rsu_matches
        )
    )
    uav_service_weight_sum = float(
        sum(
            float(edge.weight)
            for edge in uav_matches
        )
    )
    uav_net_weight_sum = float(
        uav_service_weight_sum
        - sum(
            float(
                hiring_costs[
                    provider
                ]
            )
            for provider
            in hired_uavs
        )
    )

    return MatchingSelection(
        action=_copy_action(
            chosen_action
        ),
        predicted_round_cost=float(
            chosen_cost
        ),

        baseline_cost=float(
            baseline_cost
        ),
        rsu_only_cost=float(
            rsu_only_cost
        ),
        provisional_final_cost=float(
            provisional_final_cost
        ),
        chosen_stage=str(
            chosen_stage
        ),

        rsu_candidate_edges=int(
            len(rsu_edges)
        ),
        rsu_positive_candidate_edges=int(
            rsu_positive_candidate_edges
        ),
        best_rsu_edge_weight=float(
            best_rsu_edge_weight
        ),
        rsu_matches=tuple(
            rsu_matches
        ),
        rsu_weight_sum=float(
            rsu_weight_sum
        ),

        uav_candidate_edges=int(
            len(uav_edges)
        ),
        uav_positive_candidate_edges=int(
            uav_positive_candidate_edges
        ),
        best_uav_edge_weight=float(
            best_uav_edge_weight
        ),

        provisional_uav_match_count=int(
            len(
                provisional_uav_matches
            )
        ),
        provisional_uav_provider_count=int(
            len(
                provisional_uav_providers
            )
        ),
        provisional_uav_service_weight_sum=float(
            provisional_uav_service_weight_sum
        ),
        provisional_uav_hiring_cost_sum=float(
            provisional_uav_hiring_cost_sum
        ),
        provisional_uav_net_weight_sum=float(
            provisional_uav_net_weight_sum
        ),
        best_uav_provider_net_gain=float(
            best_uav_provider_net_gain
        ),

        uav_matches=tuple(
            uav_matches
        ),
        uav_service_weight_sum=float(
            uav_service_weight_sum
        ),
        uav_net_weight_sum=float(
            uav_net_weight_sum
        ),
        hired_uavs=tuple(
            int(value)
            for value
            in hired_uavs
        ),
    )