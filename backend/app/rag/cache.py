"""Semantic cache in Redis.

Entries are stored per policy filter (the sorted policy codes, or "*" for all policies). A lookup
compares the new query embedding with the cached ones of the SAME filter and hits when the cosine
similarity reaches CACHE_SIM_THRESHOLD.

Redis layout (prefix = CACHE_PREFIX):
    {p}:entry:{id}          JSON {embedding, result}            TTL = CACHE_TTL_SECONDS
    {p}:idx:{filter}        sorted set of entry ids by time     (per-filter index, trimmed)
    {p}:policy:{code}       set of entry ids that used that policy (for invalidation)
"""

import json
import logging
import time
import uuid
from collections.abc import Sequence

import numpy as np
from redis.asyncio import Redis

from app.rag.config import RagSettings
from app.rag.schemas import PolicyAnswerResult

logger = logging.getLogger(__name__)

ALL_POLICIES = "*"


def filter_key(policy_codes: Sequence[str] | None) -> str:
    return ",".join(sorted(set(policy_codes))) if policy_codes else ALL_POLICIES


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / denominator if denominator else 0.0


class SemanticCache:
    def __init__(self, redis: Redis, settings: RagSettings):
        self.redis = redis
        self.settings = settings

    def _entry(self, entry_id: str) -> str:
        return f"{self.settings.cache_prefix}:entry:{entry_id}"

    def _index(self, key: str) -> str:
        return f"{self.settings.cache_prefix}:idx:{key}"

    def _policy(self, code: str) -> str:
        return f"{self.settings.cache_prefix}:policy:{code}"

    async def lookup(
        self, policy_codes: Sequence[str] | None, embedding: Sequence[float]
    ) -> PolicyAnswerResult | None:
        index = self._index(filter_key(policy_codes))
        ids = await self.redis.zrevrange(index, 0, self.settings.cache_max_entries_per_filter - 1)
        if not ids:
            return None
        payloads = await self.redis.mget([self._entry(i) for i in ids])

        query = np.asarray(embedding, dtype=np.float32)
        best_score, best_result = -1.0, None
        expired: list[str] = []
        for entry_id, payload in zip(ids, payloads, strict=True):
            if payload is None:  # TTL ran out (or invalidated): drop it from the index
                expired.append(entry_id)
                continue
            entry = json.loads(payload)
            score = cosine(query, np.asarray(entry["embedding"], dtype=np.float32))
            if score > best_score:
                best_score, best_result = score, entry["result"]
        if expired:
            await self.redis.zrem(index, *expired)

        if best_result is not None and best_score >= self.settings.cache_sim_threshold:
            logger.info("cache_hit", extra={"similarity": round(best_score, 4)})
            return PolicyAnswerResult.model_validate(best_result)
        return None

    async def store(
        self,
        policy_codes: Sequence[str] | None,
        embedding: Sequence[float],
        result: PolicyAnswerResult,
    ) -> None:
        key = filter_key(policy_codes)
        index = self._index(key)
        ttl = self.settings.cache_ttl_seconds
        entry_id = uuid.uuid4().hex
        payload = json.dumps(
            {
                "embedding": [round(float(x), 6) for x in embedding],
                "result": result.model_dump(mode="json"),
            }
        )

        pipe = self.redis.pipeline()
        pipe.set(self._entry(entry_id), payload, ex=ttl)
        pipe.zadd(index, {entry_id: time.time()})
        pipe.expire(index, ttl)
        for code in policy_codes or []:
            pipe.sadd(self._policy(code), entry_id)
            pipe.expire(self._policy(code), ttl)
        await pipe.execute()

        overflow = await self.redis.zcard(index) - self.settings.cache_max_entries_per_filter
        if overflow > 0:  # keep the newest entries only
            oldest = await self.redis.zrange(index, 0, overflow - 1)
            await self.redis.zrem(index, *oldest)
            await self.redis.delete(*[self._entry(i) for i in oldest])

    async def invalidate_policy(self, policy_code: str) -> int:
        """Drop every entry that could contain this policy's text: entries filtered to it, and
        entries for the 'all policies' filter. Returns how many entries were removed."""
        all_index = self._index(ALL_POLICIES)
        ids = set(await self.redis.smembers(self._policy(policy_code)))
        everywhere = await self.redis.zrange(all_index, 0, -1)
        ids.update(everywhere)
        if ids:
            await self.redis.delete(*[self._entry(i) for i in ids])
        if everywhere:
            await self.redis.zrem(all_index, *everywhere)
        await self.redis.delete(self._policy(policy_code))
        logger.info("cache_invalidated", extra={"policy": policy_code, "entries": len(ids)})
        return len(ids)
