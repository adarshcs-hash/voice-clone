# Contributing

```bash
make install-dev
make check          # ruff, ruff format, mypy --strict, pytest
```

`make check` is exactly what CI runs. Everything must be clean.

## Ground rules

**Changing a linguistic table is changing a claim about Malayalam.** The tables
in `text/numbers.py`, `text/g2p.py`, `text/translit.py` and `text/expand.py`
are this project's ground truth. Add a test case that pins the new behaviour,
say in the pull request which authority or native-speaker judgement supports it,
and read [`docs/malayalam-linguistics.md`](docs/malayalam-linguistics.md) first.

**The text frontend takes no model dependency.** `mlvoice.text` must keep
importing nothing heavier than the standard library. It is the most valuable
component precisely because it can be developed, tested and shipped without a
GPU.

**Layers stay acyclic.** Shared capabilities (ASR, speaker verification, MOS)
are protocols in `protocols.py`. If you find yourself importing `voices` from
`eval`, add a protocol instead.

**The fast suite needs no weights and no network.** Anything that needs model
weights is marked `slow` and excluded from the default run. New optional
dependencies go behind an extra and are imported lazily.

**Safety invariants are tests, not conventions.** Consent verification,
revocation cascading, deletion removing audio, tenant isolation, production
config refusal — each has a test. If you touch one, the test should be the first
thing you change.

## Adding a synthesis backend

Implement `mlvoice.tts.base.Synthesizer` — `info`, `load`, `is_ready` and
`_synthesize_chunk` — and register it:

```python
from mlvoice.tts.registry import register_backend
register_backend("my-backend", lambda settings: MyBackend(...))
```

Chunk iteration, pause insertion, concatenation and timing are handled by the
base class, so every backend behaves identically at the seams.

## Extending the test set

Add cases to `eval/testset.py`. Ids are stable and referenced by report
history — add new ones, never renumber. Set `expected_text` wherever the
frontend has exactly one correct output; those become automatic gates.

## Commit and pull-request style

One logical change per commit. In the pull request, say what behaviour changed
and how you know — a test name is a better answer than a description.
