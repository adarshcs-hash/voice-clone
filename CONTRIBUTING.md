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

## Changing the web client

`src/mlvoice/api/static/` is the whole client: one HTML file and one script,
no build step. Two rules keep it honest.

Element ids are a contract. `tests/unit/test_ui.py` checks that every id the
script looks up exists in the page, that the fields it posts are the ones the
endpoints accept, and that the response fields it reads are in the schemas — a
rename that breaks any of those fails there rather than as a button that does
nothing.

Never write caller text through `innerHTML`. The preview echoes what the user
typed back into the page; `textContent` and DOM construction only.

Anything that takes more than a moment needs a spinner, a bar and a running
seconds counter, and must clear all three on every exit path including early
returns — a bar that outlives its operation says "still working" about
something that has stopped. Test those against the slow-backend fixture in
`tests/browser/`, not the dummy one: racing a millisecond response makes the
assertion pass or fail by scheduling luck.

Run `make test-browser` for anything behavioural. It skips without a browser,
so a green `make check` is not evidence that a handler works.

## Commit and pull-request style

One logical change per commit. In the pull request, say what behaviour changed
and how you know — a test name is a better answer than a description.
