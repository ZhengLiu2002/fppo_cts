"""Shared CTS runtime contracts for algorithm implementations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CTSRuntimeContract:
    """Declarative runner-facing CTS integration options."""

    inject_constraint_names: bool = False


def resolve_cts_runtime_contract(alg_class) -> CTSRuntimeContract:
    """Return the CTS runtime contract declared by an algorithm class."""

    contract = getattr(alg_class, "cts_runtime_contract", None)
    if contract is None:
        return CTSRuntimeContract()
    if not isinstance(contract, CTSRuntimeContract):
        raise TypeError(
            f"{alg_class.__name__}.cts_runtime_contract must be a CTSRuntimeContract, "
            f"got {type(contract).__name__}."
        )
    return contract


__all__ = ["CTSRuntimeContract", "resolve_cts_runtime_contract"]
