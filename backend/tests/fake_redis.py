"""A tiny in-memory stand-in for the async Redis calls the auth / limit code uses (with a clock
tests can move), so no Redis server is needed."""

from typing import Any


class FakeRedis:
    def __init__(self) -> None:
        self.now = 1_000.0
        self.values: dict[str, Any] = {}
        self.expiry: dict[str, float] = {}
        self.hashes: dict[str, dict[str, Any]] = {}

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def _alive(self, key: str) -> bool:
        if key in self.expiry and self.expiry[key] <= self.now:
            self.values.pop(key, None)
            self.hashes.pop(key, None)
            del self.expiry[key]
        return key in self.values or key in self.hashes

    async def exists(self, *keys: str) -> int:
        return sum(1 for key in keys if self._alive(key))

    async def get(self, key: str) -> bytes | None:
        return str(self.values[key]).encode() if self._alive(key) and key in self.values else None

    async def set(
        self, key: str, value: Any, ex: int | None = None, nx: bool = False
    ) -> bool | None:
        if nx and self._alive(key):
            return None
        self.values[key] = value
        if ex is not None:
            self.expiry[key] = self.now + ex
        return True

    async def incr(self, key: str) -> int:
        self._alive(key)
        self.values[key] = int(self.values.get(key, 0)) + 1
        return self.values[key]

    async def expire(self, key: str, seconds: int) -> bool:
        self.expiry[key] = self.now + seconds
        return True

    async def ttl(self, key: str) -> int:
        if not self._alive(key):
            return -2
        return int(self.expiry[key] - self.now) if key in self.expiry else -1

    async def delete(self, *keys: str) -> int:
        for key in keys:
            self.values.pop(key, None)
            self.hashes.pop(key, None)
            self.expiry.pop(key, None)
        return len(keys)

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        if not self._alive(key):
            return {}
        return {k.encode(): str(v).encode() for k, v in self.hashes[key].items()}
