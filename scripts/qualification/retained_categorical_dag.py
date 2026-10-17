#!/usr/bin/env python3
"""Source-only conditional categorical DAG and disjoint shard contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Callable, Mapping


class OutcomeState(str, Enum):
    STRUCTURAL_CANDIDATE = "structural_candidate"
    STRICT_LOADER_REJECTED = "strict_loader_rejected"
    POST_LOAD_REJECTED = "post_load_rejected"
    AUTHORITATIVE_SCIENCE_INCOMPATIBLE = "authoritative_science_incompatible"
    RETAINED_GUARDED_OR_UNIMPLEMENTED = "retained_guarded_or_unimplemented"
    PENDING_CROSS_OWNER_REVIEW = "pending_cross_owner_review"


def compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value: object) -> str:
    return hashlib.sha256(compact(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, value: object) -> str:
    return prefix + ":" + digest(value)[:24]


@dataclass(frozen=True)
class Choice:
    value: str
    raw_spellings: tuple[str, ...]
    authority: str

    def validate(self) -> None:
        if not self.value or not self.raw_spellings or not self.authority or \
                self.value not in self.raw_spellings or \
                len(set(self.raw_spellings)) != len(self.raw_spellings):
            raise ValueError("categorical choice lacks a canonical spelling or authority")


@dataclass(frozen=True)
class Condition:
    axis: str
    equals: str

    def evaluate(self, selected: Mapping[str, str]) -> bool:
        if self.axis not in selected:
            raise ValueError("activation dependency is not earlier in the DAG: " + self.axis)
        return selected[self.axis] == self.equals


@dataclass(frozen=True)
class Axis:
    key: str
    choices: tuple[Choice, ...]
    authority: str
    active_when: tuple[Condition, ...] = ()
    inactive_reason: str = ""

    def validate(self, earlier: set[str]) -> None:
        if not self.key or not self.authority or not self.choices or self.key in earlier:
            raise ValueError("invalid or duplicate categorical axis")
        if self.active_when and not self.inactive_reason:
            raise ValueError("conditional axis lacks inactive-owner provenance")
        if any(condition.axis not in earlier for condition in self.active_when):
            raise ValueError("activation dependency is not earlier in the DAG")
        values = []
        raw = []
        for choice in self.choices:
            choice.validate()
            values.append(choice.value)
            raw.extend(choice.raw_spellings)
        if len(set(values)) != len(values) or len(set(raw)) != len(raw):
            raise ValueError("duplicate canonical value or raw spelling")

    def active(self, selected: Mapping[str, str]) -> bool:
        return all(condition.evaluate(selected) for condition in self.active_when)


def enumerate_assignments(axes: tuple[Axis, ...]) -> list[dict[str, object]]:
    earlier: set[str] = set()
    for axis in axes:
        axis.validate(earlier)
        earlier.add(axis.key)
    leaves: list[dict[str, object]] = []

    def visit(index: int, selected: dict[str, str], inactive: list[dict[str, str]]) -> None:
        if index == len(axes):
            leaves.append({"selected": dict(selected), "inactive": list(inactive)})
            return
        axis = axes[index]
        if not axis.active(selected):
            inactive.append({"axis": axis.key, "reason": axis.inactive_reason,
                             "authority": axis.authority})
            visit(index + 1, selected, inactive)
            inactive.pop()
            return
        for choice in axis.choices:
            selected[axis.key] = choice.value
            visit(index + 1, selected, inactive)
            del selected[axis.key]

    visit(0, {}, [])
    return leaves


def alias_links(axes: tuple[Axis, ...]) -> list[dict[str, str]]:
    links = []
    for axis in axes:
        for choice in axis.choices:
            for raw in choice.raw_spellings:
                if raw != choice.value:
                    links.append({"axis": axis.key, "raw": raw,
                                  "canonical": choice.value, "authority": choice.authority})
    return sorted(links, key=compact)


def negative_intent(axis: str, raw: str, context: dict[str, str],
                    outcome: OutcomeState, reason: str, authority: str,
                    next_task: str = "", observed_admission: str = "") -> dict[str, object]:
    if outcome == OutcomeState.STRUCTURAL_CANDIDATE or not axis or not raw or \
            not reason or not authority or \
            (outcome == OutcomeState.RETAINED_GUARDED_OR_UNIMPLEMENTED and not next_task):
        raise ValueError("negative retained intent lacks disposition authority")
    content = {"axis": axis, "raw": raw, "context": context,
               "outcome": outcome.value}
    return {"id": stable_id("raw-categorical-intent", content), **content,
            "reason": reason, "authority": authority,
            "nextTask": next_task, "observedAdmission": observed_admission}


def leaf(signature: dict[str, object], selected: dict[str, str],
         inactive: list[dict[str, str]], outcome: OutcomeState,
         evidence_class: str, reason: str = "", authority: str = "") -> dict[str, object]:
    if not evidence_class or \
            (outcome != OutcomeState.STRUCTURAL_CANDIDATE and
             (not reason or not authority)):
        raise ValueError("leaf outcome lacks evidence class or rejection authority")
    return {"id": stable_id("conditional-combination", signature),
            "signature": signature, "selected": selected, "inactive": inactive,
            "outcome": outcome.value, "evidenceClass": evidence_class,
            "reason": reason, "authority": authority,
            "admission": "not_checked_by_source_only_dag",
            "physicalExecution": "not_checked", "I1Complete": False}


def shard_index(identifier: str, shard_count: int) -> int:
    if shard_count < 1 or not identifier:
        raise ValueError("invalid shard configuration")
    return int(hashlib.sha256(identifier.encode("utf-8")).hexdigest(), 16) % shard_count


def partition(leaves: list[dict[str, object]], shard_count: int) -> list[dict[str, object]]:
    if shard_count < 1:
        raise ValueError("shard count must be positive")
    ids = [row["id"] for row in leaves]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate canonical structural ID")
    shards = []
    for index in range(shard_count):
        rows = sorted((row for row in leaves if shard_index(row["id"], shard_count) == index),
                      key=lambda row: row["id"])
        shards.append({"shard": index, "count": len(rows),
                       "ids": [row["id"] for row in rows],
                       "recordsSha256": digest(rows), "records": rows})
    verify_partition(leaves, shards, shard_count)
    return shards


def verify_partition(leaves: list[dict[str, object]],
                     shards: list[dict[str, object]],
                     expected_shard_count: int | None = None) -> None:
    shard_count = expected_shard_count if expected_shard_count is not None else len(shards)
    if shard_count < 1 or [row["shard"] for row in shards] != list(range(shard_count)):
        raise ValueError("skipped, duplicate or reordered shard")
    expected = {row["id"]: row for row in leaves}
    if len(expected) != len(leaves):
        raise ValueError("duplicate source leaf ID")
    observed: dict[str, dict[str, object]] = {}
    for shard in shards:
        index = shard["shard"]
        rows = shard["records"]
        if shard["count"] != len(rows) or shard["ids"] != [row["id"] for row in rows] or \
                shard["ids"] != sorted(shard["ids"]) or \
                shard["recordsSha256"] != digest(rows):
            raise ValueError("shard count, order or record digest drifted")
        for row in rows:
            identifier = row["id"]
            if identifier in observed or identifier not in expected or \
                    shard_index(identifier, shard_count) != index or \
                    row != expected[identifier]:
                raise ValueError("shard ID, membership or source record drifted")
            observed[identifier] = row
    if set(observed) != set(expected):
        raise ValueError("deleted or orphaned structural ID")


def map_assignments(axes: tuple[Axis, ...],
                    resolver: Callable[[dict[str, str], list[dict[str, str]]], dict[str, object]]) -> list[dict[str, object]]:
    return [resolver(row["selected"], row["inactive"])
            for row in enumerate_assignments(axes)]
