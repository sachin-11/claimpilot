from enum import StrEnum

from sqlalchemy import Enum


def pg_enum(enum_cls: type[StrEnum], name: str) -> Enum:
    """Native Postgres enum that stores the lowercase values, not the member names."""
    return Enum(enum_cls, name=name, values_callable=lambda e: [m.value for m in e])
