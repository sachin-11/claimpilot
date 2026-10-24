import httpx
import openai
import pytest
from pdf_fixtures import FakeEmbedder, fake_vector, ingestion_settings, whitespace_tokens

from app.ingestion.embedder import embed_texts, make_batches


async def no_sleep(_seconds: float) -> None:
    return None


def test_make_batches_respects_item_and_token_limits() -> None:
    assert make_batches(["a"] * 5, [1] * 5, max_items=2, max_tokens=100) == [[0, 1], [2, 3], [4]]
    assert make_batches(["a"] * 4, [60, 60, 10, 10], max_items=10, max_tokens=100) == [
        [0],
        [1, 2, 3],
    ]


async def test_embeds_in_batches_and_reports_tokens_and_cost(tmp_path) -> None:
    cfg = ingestion_settings(tmp_path, embed_batch_size=64, embed_cost_per_million_tokens=0.02)
    embedder = FakeEmbedder()
    texts = [f"text number {i}" for i in range(150)]

    result = await embed_texts(texts, embedder, cfg, tokens=whitespace_tokens)

    assert [len(c) for c in embedder.calls] == [64, 64, 22]
    assert result.vectors[7] == fake_vector("text number 7")  # order preserved
    assert result.tokens_used == 450 and result.batches == 3
    assert result.estimated_cost_usd == pytest.approx(450 / 1_000_000 * 0.02)


async def test_retries_rate_limits_with_backoff_then_succeeds(tmp_path) -> None:
    cfg = ingestion_settings(tmp_path, embed_max_retries=4)

    class Flaky(FakeEmbedder):
        async def aembed_documents(self, texts):
            if len(self.calls) < 2:
                self.calls.append([])
                raise openai.APITimeoutError(request=httpx.Request("POST", "http://x"))
            return await super().aembed_documents(texts)

    embedder = Flaky()
    result = await embed_texts(["a b"], embedder, cfg, tokens=whitespace_tokens, sleep=no_sleep)

    assert len(result.vectors) == 1
    assert len(embedder.calls) == 3  # two failures + one success


async def test_gives_up_after_max_retries(tmp_path) -> None:
    cfg = ingestion_settings(tmp_path, embed_max_retries=3)

    class AlwaysTimeout(FakeEmbedder):
        async def aembed_documents(self, texts):
            self.calls.append(texts)
            raise openai.APITimeoutError(request=httpx.Request("POST", "http://x"))

    embedder = AlwaysTimeout()
    with pytest.raises(openai.APITimeoutError):
        await embed_texts(["a"], embedder, cfg, tokens=whitespace_tokens, sleep=no_sleep)
    assert len(embedder.calls) == 3


async def test_wrong_dimension_is_rejected(tmp_path) -> None:
    class SmallVectors:
        async def aembed_documents(self, texts):
            return [[0.1, 0.2] for _ in texts]

    with pytest.raises(ValueError, match="dimensions"):
        await embed_texts(["a"], SmallVectors(), ingestion_settings(tmp_path))


async def test_nothing_to_embed_makes_no_call(tmp_path) -> None:
    embedder = FakeEmbedder()
    result = await embed_texts([], embedder, ingestion_settings(tmp_path))
    assert result.vectors == [] and embedder.calls == []


async def test_exhausted_quota_is_not_retried(tmp_path) -> None:
    response = httpx.Response(429, request=httpx.Request("POST", "http://x"))

    class NoCredits(FakeEmbedder):
        async def aembed_documents(self, texts):
            self.calls.append(texts)
            raise openai.RateLimitError(
                "no credits",
                response=response,
                body={"code": "insufficient_quota", "message": "no credits"},
            )

    embedder = NoCredits()
    with pytest.raises(openai.RateLimitError):
        await embed_texts(
            ["a"], embedder, ingestion_settings(tmp_path), tokens=whitespace_tokens, sleep=no_sleep
        )
    assert len(embedder.calls) == 1  # failed fast, no backoff loop
