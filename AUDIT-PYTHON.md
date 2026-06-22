# Python Code Audit — `transcribe`

Date: 2026-06-22 · Python 3.12 · package: transcribe

Scope: 7,205 source lines across 31 modules; 3,374 test lines across 16 test files. No `pyproject.toml` (ruff/mypy run on defaults; project uses 2-space indent). All findings below were independently verified against the source before inclusion.

---

## ✓ Resolution Status — updated 2026-06-23

**Every finding in this report has been resolved.** The **Health Score: 4/10** and all
severity text below are preserved as the original 2026-06-22 audit snapshot (the
*pre-remediation* state) and are intentionally left unrewritten.

Fixes landed on `main` across these commits:

| Commit | What it fixed |
|--------|---------------|
| `68b3a74` | High quick-wins — lazy `OpenAIClient` (Claude usable without `OPENAI_API_KEY`), drop `max_workers` from `Transcriber()`, stable SHA-256 cache keys, revert `reasoning_effort` to `"minimal"`; ruff `--fix` sweep |
| `e1fa534` | Both **Critical** findings — `text.find()` newline corruption and parallel chunk-overlap duplication |
| `0d43e1c` | Dead-code removal — `ChunkProcessor`, `DEFAULT_CHUNK_OVERLAP` |
| `88cb7d7` | **All remaining findings** (5 High, ~12 Medium, ~31 Low + lint), each with a regression test (6 new test files added) |

Follow-up housekeeping: pruned phantom `__all__` entries (`ContentAwareProcessor`,
`ChunkCombiner`) from `core/__init__.py` and the stale `cached` entry from
`utils/__init__.py`.

Two resolutions worth noting (judgment calls, not literal applications of the suggested fix):
- **Specialized prompt templates** — resolved by *pruning* the undefined `speech_*`/`lecture_*`/`*_summary` template references (option b) so the code states the truth, not by adding six new prompt templates (a feature, out of scope).
- **Per-index timestamp offset** — both timestamp findings collapsed into one fix: a fixed `i * chunk_length` offset, which is provably exact because every non-final chunk is exactly `chunk_length_ms`.

**Verification at `88cb7d7`:** 250/250 unit tests pass (`.venv/bin/python -m pytest`,
excluding the two live-service suites); `ruff check .` clean. The
`test_ollama_integration` / `test_audio_integration` suites require a live Ollama with
standard models and `ANTHROPIC_API_KEY`; their in-sandbox failures are environmental,
not regressions.

---

## Executive Summary

**Health Score: 4/10**

The package is feature-rich and reasonably well-organized (clean provider abstraction, content-aware processing, caching layer, subtitle generation), but it carries several **silent data-correctness defects** on its default code paths. Three Critical / High issues independently corrupt text output: the sequential chunk reassembler garbles ordinary multi-line input via a `text.find()` on newline-normalized content, the parallel path re-emits overlapping sentences because the dedup routine is dead code, and timestamp stitching drifts cumulatively (and jumps a full chunk-width when any chunk is skipped). None of these raise an error — the user gets a successful exit code and wrong output. Compounding this, the disk cache (documented as "critical") never hits across runs because keys embed the per-process-randomized builtin `hash()`, and the documented default Claude model cannot even be constructed without an `OPENAI_API_KEY`. The score is held down by the fact that the most-used commands (`clean-transcript` on a multi-line file, `transcribe` on >10-min audio with subtitles, default parallel post-processing) each hit at least one silent-corruption bug. It is *not* a security disaster and the architecture is sound; the problems are concentrated in chunk boundary handling, timestamp math, and cache-key construction — all fixable without redesign.

**Critical Issues**
- Newline in input corrupts text and spins a runaway loop in the default sequential cleaner (`processor.py:540-544`).
- Parallel post-processing duplicates overlapping sentences at every seam; the dedup routine is never called (`parallel.py:145`, `processor.py:325`).

**Quick Wins**
- Disk cache never persists across runs — swap builtin `hash()` for a stable hash (`processor.py:156,284`). Pure cost/latency win.
- Default Claude model unusable without `OPENAI_API_KEY` — make `OpenAIClient` lazy (`api_utils.py:95-100`).
- `transcribe_audio_file()` crashes with `TypeError` on every unmocked call — drop `max_workers` from the constructor (`transcriber.py:499-502`).
- `reasoning_effort="none"` rejected by installed SDK 2.6.1 — revert to `"minimal"` (`api_utils.py:77`).
- 36 of 42 ruff violations are auto-fixable (`ruff check --fix`).

**Architectural Notes**
- Chunk-overlap context strategy is half-built: `split_text_for_processing` creates overlap, but `ChunkProcessor.adjust_chunk_boundaries()` / `combine_chunks()` (the intended dedup) have zero callers. Either wire them in or set `overlap=0`.
- Timestamp offset is reconstructed from `segments[-1].end` (last spoken word) instead of true chunk duration — fragile by design; switch to a per-index fixed offset.
- Two `ParallelProcessor` classes with different APIs coexist (`core/parallel.py` vs `utils/progress.py`); `use_processes=True` is a public option that cannot work (unpicklable closure). Both are maintenance traps.
- Cache-key construction is duplicated across three findings and is the single highest-leverage refactor: centralize hashing and include all output-affecting params (provider, max_tokens, prompt identity).

**Deduplication note:** The 46 raw findings reduced to **34 distinct issues**. Twelve were merged as duplicates of the same root cause/location:
- **Chunk-overlap duplication** — 3 findings merged into one (Critical) (`postprocess-core`, `concurrency`, `dataflow` all reported `processor.py:117/325` + `parallel.py:145`).
- **`_get_chunk_with_complete_sentences` `text.find()` defect** — 2 findings merged into one Critical (`dataflow` newline-corruption + `postprocess-core` fragile-locate are the same code at `processor.py:540-544`; the Critical framing subsumes the Medium one).
- **Non-deterministic `hash()` cache keys** — 3 findings merged into one (`infra`, `concurrency`, `dataflow`, all `processor.py:156,284`).
- **Thread-unsafe shared `CacheManager`** — 2 findings merged into one (`postprocess-core` + `concurrency`, both `cache.py` memory eviction under `processor.py` workers).
- **Free-function `transcribe_audio()` `.audio` AttributeError** — 2 findings merged into one (`llm-providers` + `errors-resources`, both `api_utils.py:495-497`).
- **`AudioProcessor` mkdtemp leak** — 2 findings merged into one (`transcribe-core` + `errors-resources`, both `audio_utils.py:148` vs `58-67`).
- **Subtitle empty-cue / word-sparse split** — 2 findings merged into one (`infra` + `dataflow`, both `subtitle_utils.py:86-115` / `171-198`).

---

## Automated Tooling

**ruff — 42 violations (36 auto-fixable via `ruff check --fix`):**

| Code | Count | Category |
|------|-------|----------|
| F401 | 28 | unused-import (stale imports across multiple modules) |
| F541 | 8 | f-string without placeholders (should be plain strings) |
| F841 | 3 | assigned-but-unused local variable |
| E402 | 1 | module-import-not-at-top-of-file |
| E722 | 1 | bare `except:` |
| E741 | 1 | ambiguous variable name |

The bare-except (E722) and the unused locals (F841) overlap with manual findings (e.g. the `clear()` bare-except at `cache.py:207`, flagged in the cache-tmp leak finding). None of the manual correctness bugs are auto-flagged by ruff — they are module-level defs (dead code), control-flow logic, or per-process hashing, all invisible to lint.

**mypy — 54 type errors** across `config.py`, `text_utils.py`, `cache.py`, `audio_utils.py`, `progress.py`, `parallel.py`, `openai_client.py`, `api_utils.py`. Top categories:

1. **JSON/dict indexing on untyped objects** — `config.py` alone has 11 ("Unsupported target for indexed assignment", "Value of type object is not indexable"). The JSON config shape is untyped; a `TypedDict` (or explicit `dict[str, Any]` casts) is needed. This is the same `config.py` that carries the shallow-copy and precedence bugs below — the section is under-typed end to end.
2. **Missing variable annotations** — 5+ `var-annotated` errors (`current_words`, `memory_cache`, `temp_files`, `processed_chunks`) needing `list[T]` / `dict[K, V]`.
3. **OpenAI API signature mismatches** — 5+ errors: `messages` argument typed as `list[dict[str, str]]` where the SDK expects `ChatCompletionUserMessageParam` et al. (manual message-dict construction), plus `transcribe`/`create` overload mismatches. This corroborates the manual finding that `reasoning_effort="none"` and the message construction are out of step with the installed SDK.

**Refactor diff (uncommitted, 3 files / +21 −11):** `_default_reasoning_effort()` was cleanly extracted into `api_utils.py` and wired through `analyzer.py` and `prompts.py` (replacing inline `_is_reasoning_model()`). The extraction itself is good. **However the same diff introduced the `reasoning_effort="none"` regression** (see Critical/High and Findings) — `"none"` is not in the installed SDK 2.6.1 enum, and the docstrings now advertise `"none"`/`"xhigh"` the SDK does not support. The refactor is structurally sound but functionally broke OpenAI reasoning-model aux calls.

---

## Findings

Grouped by severity. Each block: **Title** · Location · Category · Description · Impact · Fix.

### Critical

---

**Newline in input corrupts text and triggers runaway loop in `_get_chunk_with_complete_sentences`**
Location: `transcribe_pkg/core/processor.py:540-544`
Category: Bug *(merges 2 findings: dataflow newline-corruption + postprocess-core fragile-locate)*

Description: The chunk is built by joining sentences from `create_sentences()`, which normalizes newlines to spaces (`text_utils.py:59` `sentence.replace('\n',' ')`). The remaining text is then located in the ORIGINAL text via `text.find(clean_chunk)`. When the consumed span contains a newline, the normalized `clean_chunk` no longer exists in `text`, so `find()` returns `-1`; `chunk_position = -1 + len(clean_chunk)` lands mid-text and `remaining_text` becomes a garbage tail (e.g. `'.'`). The default `_process_sequential` loop then spins re-processing junk until the iteration-limit safety net fires.

Impact: `clean-transcript` reads files with `file.read()` (newlines intact) and runs the sequential path by default; `transcribe`'s default post-processing (no `--parallel`) does too. Reproduced: `'First sentence here.\nSecond sentence here. Third one.'` → `find()==-1`, `remaining_text=='.'`; a multi-chunk newline-joined transcript hit the 100-iteration guard producing 99 garbage chunks vs 95 original words. Garbled/duplicated/truncated output plus tens of wasted paid LLM calls on the normal multi-line case.

```python
  # Track consumption by length, not by re-finding normalized content.
  clean_chunk = chunk.strip()
  idx = text.find(clean_chunk)
  if idx == -1:
    # normalized chunk not present verbatim (newlines collapsed to spaces);
    # fall back to slicing off exactly what we consumed to guarantee progress
    remaining_text = text[len(chunk):].lstrip()
  else:
    chunk_position = idx + len(clean_chunk)
    remaining_text = text[chunk_position:] if chunk_position < len(text) else ''
  return clean_chunk, remaining_text
  # Robust long-term fix: have create_sentences() return (sentence, end_offset)
  # and consume remaining_text by offset rather than string search.
```

---

**Chunk overlap is added but never removed at reassembly → duplicated text at every parallel seam**
Location: `transcribe_pkg/core/parallel.py:145` (and `processor.py:117,325`; overlap source `text_utils.py:201-219`)
Category: LogicFlow *(merges 3 findings: postprocess-core, concurrency, dataflow)*

Description: `TranscriptProcessor` builds its `ParallelProcessor` with `overlap=500` (`processor.py:117`), so `split_text_for_processing` makes adjacent chunks share trailing/leading sentences for context. But reassembly is a plain `"\n\n".join(...)` — `combine_func` at `processor.py:325` and the default branch at `parallel.py:145` — with no dedup. Every overlapping sentence appears twice. The routine written to strip overlap, `ChunkProcessor.adjust_chunk_boundaries()` (`parallel.py:295`) and `combine_chunks()` (`parallel.py:250`), has zero callers (grep-confirmed dead code).

Impact: `process()` defaults `use_parallel=True`; any document larger than `max_chunk_size` (default 3000 bytes) routes to `_process_parallel`. Output contains repeated sentences at every seam after the first (the `len(chunks) > 1` guard skips the first seam). Visibly corrupted output that scales with document length. Reproduced empirically: `split_text_for_processing(text, max_chunk_size=120, overlap=40)` yielded duplicate sentences at every subsequent seam.

```python
  # Simplest correct fix: stop overlapping the emitted chunks. Overlap adds no
  # benefit when chunks are cleaned independently, and the reassembler can't dedup.
  self.parallel_processor = ParallelProcessor(
    max_workers=max_workers,
    use_processes=False,
    chunk_size=max_chunk_size,
    overlap=0,  # chunks processed independently; overlap only duplicates on recombine
    logger=self.logger,
  )
  # If cross-chunk context is wanted, feed the previous chunk's tail as a separate
  # prev_context system-prompt input rather than emitting it as duplicated content.
```

### High

---

**Default Claude model unusable without `OPENAI_API_KEY` (OpenAIClient constructor hard-requires the key)**
Location: `transcribe_pkg/utils/api_utils.py:95-100` (triggered via `processor.py:75` + `cli/commands.py:641`)
Category: Bug

Description: `OpenAIClient.__init__` raises `ValueError` when `OPENAI_API_KEY` is unset and eagerly builds `openai.OpenAI(...)`. `TranscriptProcessor.__init__` (and `ContentAnalyzer`/`Transcriber`/etc.) unconditionally do `self.api_client = api_client or OpenAIClient(...)`, and `clean_transcript_command` never injects one. So constructing the processor for ANY model — including the default `claude-sonnet-4-5` — requires an OpenAI key, even though no OpenAI/Whisper call is made.

Impact: `clean-transcript raw.txt` aborts with "OpenAI API key is required" for any user who configured only `ANTHROPIC_API_KEY`/`GOOGLE_API_KEY`/Ollama. The headline multi-provider feature fails out of the box. Reproduced: with only `ANTHROPIC_API_KEY` set, `TranscriptProcessor(model='claude-sonnet-4-5')` raises `ValueError`.

```python
  def __init__(self, api_key=None, logger=None):
    self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
    self.logger = logger or get_logger(__name__)
    self._client = None
  @property
  def client(self):
    if self._client is None:
      if not self.api_key:
        raise ValueError("OpenAI API key is required. ...")
      self._client = openai.OpenAI(api_key=self.api_key)
    return self._client
```

---

**`transcribe_audio_file()` passes `max_workers` to `Transcriber()`, which does not accept it → guaranteed TypeError**
Location: `transcribe_pkg/core/transcriber.py:499-502`
Category: Bug

Description: The exported high-level helper constructs `Transcriber(language=..., max_workers=...)`, but `Transcriber.__init__` (`transcriber.py:30-39`) has no `max_workers` parameter (`max_workers` belongs on `transcribe()`). The TypeError fires at construction, before the function's only `try/except`. The end-to-end test masks it by patching out the entire `Transcriber` class (`tests/test_integration.py:33`).

Impact: Every real (unmocked) caller of the documented high-level API crashes immediately with `TypeError: Transcriber.__init__() got an unexpected keyword argument 'max_workers'`. Verified at runtime via `inspect.signature`. The keyword is passed unconditionally, so both parallel and non-parallel paths crash.

```python
  transcriber = Transcriber(language=language)
  transcript_result = transcriber.transcribe(
    audio_path=audio_path,
    prompt=context,
    with_timestamps=with_timestamps,
    max_workers=max_workers if parallel_processing else 1,
  )
```

---

**Disk cache never persists across runs: cache key uses non-deterministic builtin `hash()`**
Location: `transcribe_pkg/core/processor.py:156,284`
Category: Bug *(merges 3 findings: infra, concurrency, dataflow)*

Description: Keys embed Python's builtin `hash()` of strings: `f"language_detection:{hash(sample_text)}"` and `f"chunk_processing:{hash(chunk)}:{hash(chunk_context)}:{language}:{self.model}:{self.temperature}"`. Python randomizes string hashing per process (`PYTHONHASHSEED` unset), so the key differs every invocation; `CacheManager._hash_key` then md5's the already-randomized string, producing a new disk filename each run. Secondary defect: the chunk key omits `self.max_tokens` and `self.provider` (both passed to `call_llm`), so changing `--provider`/`--max-tokens` within a run can serve a stale result.

Impact: Disk cache (`~/.cache/transcribe`, on by default) NEVER hits across separate `transcribe`/`clean-transcript` invocations — every re-run re-pays for all LLM calls, the opposite of the documented "caching is critical" contract. Verified: two `python3 -c 'print(hash("abc"))'` runs returned different values. Memory cache still works within a process, masking it in tests.

```python
  import hashlib
  def _stable(*parts: object) -> str:
    h = hashlib.sha256()
    for p in parts:
      h.update(repr(p).encode()); h.update(b'\x00')
    return h.hexdigest()
  cache_key = f"language_detection:{_stable(sample_text, self.summary_model)}"
  cache_key = (
    f"chunk_processing:"
    f"{_stable(chunk, chunk_context, language, self.model, self.temperature, self.max_tokens, self.provider)}"
  )
```

---

**`DEFAULT_CONFIG.copy()` is a shallow copy: nested `set()`/`update()` permanently mutate module-global defaults**
Location: `transcribe_pkg/utils/config.py:91`
Category: Bug

Description: `Config.__init__` does `self._config = DEFAULT_CONFIG.copy()`. `dict.copy()` is shallow, so every nested dict (`openai`, `openai.models`, `transcription`, `processing`, …) is SHARED with the module-global `DEFAULT_CONFIG`. `set()` (`config.py:277`) and `_update_nested_dict()` mutate those in place, rewriting the canonical defaults. Worse, `load_from_env()` runs in every `__init__` and writes nested values in place — so merely constructing `Config()` with `OPENAI_API_KEY` set writes the live secret into `DEFAULT_CONFIG['openai']['api_key']`.

Impact: Config state leaks across instances and corrupts the canonical defaults for the process lifetime; a `Config` built after any CLI parse no longer reflects the real defaults; multi-instance tests become order-dependent. Reproduced: `c.set('openai.models.completion','MUTATED')` mutates `DEFAULT_CONFIG`, and a fresh `Config()` then returns `'MUTATED'`. The module docstring's "Deep-copy protection" claim is false.

```python
  import copy
  # ...
  self._config = copy.deepcopy(DEFAULT_CONFIG)
```

---

**Cross-chunk timestamp offset uses last-segment end instead of chunk duration → cumulative drift**
Location: `transcribe_pkg/core/transcriber.py:273-279` (and identical logic at `450-456`)
Category: Bug

Description: When stitching per-chunk Whisper segments into a global timeline, `total_duration` is advanced by the END timestamp of the chunk's LAST SEGMENT, not the chunk's actual audio length. Each chunk is exported as an independent file starting at t=0, so the offset for chunk N must be the sum of real chunk durations. The last segment's `.end` is where the last speech/VAD activity ends — almost always less than the chunk length (trailing silence/music never counted) — so each chunk under-shifts the next, and the error accumulates.

Impact: For any multi-chunk file (default `chunk_length_ms=600000` → any audio >10 min) transcribed with `--timestamps`/`--srt`/`--vtt`, subtitle timing drifts progressively out of sync. Silently wrong (no exception) — and this is the exact feature timestamps exist for.

```python
  # Replace the running accumulator with a fixed per-index offset based on
  # real chunk duration. Drop the segments[-1].end logic entirely.
  total_duration += self.chunk_length_ms / 1000.0
  # Best: compute offset_for_chunk_i = i * (self.chunk_length_ms / 1000.0)
  # (thread info.duration for the final, shorter chunk). Apply at both ~279 and ~456.
```

---

**Skipped/failed chunk does not advance timestamp offset → full-chunk-width jump for all later timestamps**
Location: `transcribe_pkg/core/transcriber.py:378-456` (parallel) and `196-279` (sequential)
Category: LogicFlow

Description: In timestamp mode, an empty/failed chunk hits `continue` BEFORE the `total_duration += ...` update (the update lives only in the per-chunk success branch, guarded by `if segments:`). So one skipped chunk omits its entire duration from the offset, shifting every subsequent segment earlier. In the parallel path the failed future stored as `{"text":"","segments":[]}` is caught at the empty-text branch, still skipping the offset update.

Impact: One transient API failure or one all-silence chunk in the middle of a long file desynchronises every subsequent subtitle by ~the full chunk length (up to 10 min with defaults). Silent corruption, compounding the cumulative-drift finding above.

```python
  # Use a fixed per-index offset so the offset is independent of chunk contents
  # and cannot be skipped by `continue`:
  offset_for_chunk_i = i * (self.chunk_length_ms / 1000.0)
  # apply offset_for_chunk_i to each segment; remove the running `total_duration`
  # accumulator and its `if segments:` guarded update in both paths.
```

---

**Parallel post-processing glues words / sequential reassembly fuses chunks lacking terminal punctuation**
Location: `transcribe_pkg/core/processor.py:406-408`
Category: Bug

Description: When stitching processed chunks in `_process_sequential`, a separating space is inserted only if the previous text ends in `.,?!` backtick `"`: `if generated_text and generated_text[-1] in '.,?!`"': generated_text += ' '`. LLM-cleaned chunks frequently end on a letter, digit, `)`, `:`, or a single quote `'` (NOT in the set). In those cases the next chunk is concatenated with no separator, fusing the last word of one chunk to the first word of the next. The closing-quote handling is also asymmetric (`"` is whitelisted, `'` is not).

Impact: `_process_sequential` is the DEFAULT path (`--parallel` defaults off; `clean-transcript` hardcodes `use_parallel=False`) for any multi-chunk transcript (>3000 bytes). Word boundaries are lost at every seam ending in a non-whitelisted, non-whitespace char. Reproduced: `'This is the end of chunk one'` + `'And here is chunk two.'` → `'...oneAnd here...'`.

```python
  if generated_text and not generated_text.endswith((' ', '\n')):
    generated_text += ' '
  generated_text += processed_chunk
```

---

**`-T/--timestamps` with file output silently drops all timestamp information**
Location: `transcribe_pkg/cli/commands.py:328-355`
Category: Bug

Description: For stdout, `_write_output` emits `[start -> end] text` per segment. For FILE output it has only two branches: subtitle generation (`with_timestamps AND subtitle_format`) or the `else` plain-text write. When `-T` is given without `--srt`/`--vtt` and an output file, `subtitle_format` is None, so control falls to `else`, which writes `_extract_text_from_result(result)` — plain text only. The per-segment timing captured by `verbose_json` is discarded.

Impact: `transcribe audio.mp3 -T -o out.txt` produces a plain-text file with no timestamps, silently contradicting the flag's documented contract. The expensive verbose transcription work is computed and thrown away; no error, no timestamps.

```python
  else:
    logger.info(f"Writing transcript to {output_path}")
    if with_timestamps:
      segments = result.get("segments", []) if isinstance(result, dict) else []
      if segments:
        lines = [f"[{s.get('start', 0):.2f} -> {s.get('end', 0):.2f}] {s.get('text', '').strip()}"
                 for s in segments]
        text_content = "\n".join(lines)
      else:
        logger.warning("No segments found; writing plain text")
        text_content = _extract_text_from_result(result)
    else:
      text_content = _extract_text_from_result(result)
    with open(output_path, "w", encoding="utf-8") as f:
      f.write(text_content)
```

### Medium

---

**`reasoning_effort="none"` is not valid for the installed OpenAI SDK (uncommitted edit broke it)**
Location: `transcribe_pkg/utils/api_utils.py:77`
Category: Bug

Description: The new `_default_reasoning_effort()` returns `"none"` for reasoning models (previously `"minimal"`). Installed `openai==2.6.1` defines `ReasoningEffort = Optional[Literal["minimal","low","medium","high"]]` — `"none"` is not in the enum, so the API returns 400. It is forwarded unfiltered at `api_utils.py:268` (`"none"` is truthy). New docstrings also advertise `"none"`/`"xhigh"` the SDK does not support.

Impact: Whenever an OpenAI reasoning model (gpt-5*/o1*/o3*) is the analyzer/summary model, every content-type detection, context extraction, and language detection raises 400, swallowed by broad `except Exception` handlers and degraded to fallback values ('general', no domains, 'en'). Latent until someone selects an OpenAI reasoning model (defaults are Claude).

```python
  def _default_reasoning_effort(model: str) -> str | None:
    return "minimal" if _is_reasoning_model(model) else None
  # and revert the docstrings at api_utils.py:235,350 to list "minimal","low","medium","high".
  # If "none"/"xhigh" is genuinely targeted later, gate it behind an SDK/version check.
```

---

**OpenAIClient.chat_completion crashes on None content from reasoning models instead of returning `''`**
Location: `transcribe_pkg/utils/api_utils.py:281` (and `:290`)
Category: Bug

Description: `ChatCompletionMessage.content` is `Optional[str]`; reasoning models often return `content=None`. The guard `not response.choices[0].message.content.strip()` calls `.strip()` on None → `AttributeError`, caught by broad `except Exception` and re-raised as `APIError` (not in the retry predicate). This contradicts the code's own comment ("Empty responses can happen with reasoning models … return ''").

Impact: A reasoning model returning empty content turns a recoverable empty-response into a hard `APIError` that aborts the calling step. Reproduced with a mocked `content=None`, `model='gpt-5'`. Non-default (Claude is default), but real when OpenAI reasoning models are selected.

```python
  content = response.choices[0].message.content if response.choices else None
  if not response.choices or not content or not content.strip():
    if _is_reasoning_model(model):
      self.logger.debug("Empty response from reasoning model %s", model)
    else:
      self.logger.warning("Empty response from model %s", model)
    return ""
  return content.strip()
```

---

**Anthropic client silently drops `temperature=0.0`, defaulting Claude to `temperature=1.0` for "deterministic" calls**
Location: `transcribe_pkg/utils/providers/anthropic_client.py:53-54`
Category: LogicFlow

Description: The client forwards temperature only when `temperature > 0`. Every deterministic auxiliary call passes `temperature=0.0` (content-type detection, summarize, context extraction, language detection). With 0.0 omitted, the Anthropic API applies its server default of 1.0. Since `claude-sonnet-4-5`/`claude-haiku-4-5` are the defaults, the most accuracy-sensitive calls (single-word classification, two-letter language code) run at max randomness. The OpenAI path sets temperature unconditionally, so this is also a cross-provider inconsistency.

Impact: Language detection and content classification with default Claude models are non-deterministic and more error-prone; repeated runs on identical input can yield different categories/codes. `temperature=0` is valid for Anthropic, so suppressing it is simply wrong.

```python
  create_kwargs["temperature"] = max(0.0, min(temperature, 1.0))
  # drop the `if temperature > 0` gate entirely; 0.0 is a legitimate deterministic setting
```

---

**Prompt templates referenced in `get_specialized_prompt` are never defined (speech/lecture cleaning + ALL `*_summary`)**
Location: `transcribe_pkg/core/analyzer.py:103-129`
Category: LogicFlow

Description: `template_mapping` references `speech_cleaning`, `lecture_cleaning`, `dialogue_summary`, `technical_summary`, `speech_summary`, `lecture_summary`, but `PromptManager._templates` defines only `transcript_processing`, `dialogue_cleaning`, `technical_cleaning`, `context_summary`, `context_extraction`, `language_detection`. The fallback silently substitutes the generic template, so speech/lecture content always gets generic `transcript_processing`, and EVERY content type in 'summarize' mode gets generic `context_summary`.

Impact: A core advertised feature (content-aware specialized processing) silently no-ops for speech, lecture, and all summarization paths even when classification is correct. No crash, but degraded output and a misleading code surface — exactly the "silent fallback to generic" failure the project's own CLAUDE.md warns against.

```python
  # Option (b): prune the dead mapping so the code states the truth.
  "clean": {
    "dialogue": "dialogue_cleaning",
    "technical": "technical_cleaning",
    "general": "transcript_processing",
  },
  "summarize": {"general": "context_summary"},
  # Option (a): add the missing templates to PromptManager._templates + a test per type.
```

---

**Config precedence contradicts documented priority: env vars override the JSON config file**
Location: `transcribe_pkg/utils/config.py:95-99`
Category: LogicFlow

Description: CLAUDE.md states precedence `defaults < env < json < cli` (the file should beat env). But `__init__` applies the file FIRST (`load_from_file`) then env SECOND (`load_from_env`), and `load_from_env` unconditionally overwrites file values (e.g. `OPENAI_COMPLETION_MODEL`, `TRANSCRIBE_*`). So env beats the file — the inverse of the authoritative doc. (The module's own docstring lists the opposite order, so the two docs also disagree.)

Impact: A user who pins a model/setting in their JSON config is silently overridden by a stale environment variable. Hard-to-debug "my config file is ignored" reports. Reproduced: env `OPENAI_COMPLETION_MODEL` wins over a JSON-file value.

```python
  # defaults already in self._config
  self.load_from_env()
  if config_file:
    self.load_from_file(config_file)
  # then reconcile the module docstring (lines 6-10) with CLAUDE.md — pick one order.
```

---

**`_measure_technical_level` raises ZeroDivisionError on whitespace-only text**
Location: `transcribe_pkg/core/analyzer.py:274-275`
Category: ErrorHandling

Description: `words = len(text.split())` is 0 for empty/whitespace-only text, and `score = min(1.0, term_count / (words * 0.05))` then divides by zero. `analyze_content` has no `try/except`, and `SpecializedProcessor.process_content` wraps only the `call_llm` step, so the exception propagates and crashes the whole call. (Note: the sibling `_measure_dialogue_ratio` correctly guards with `max(1, ...)`.)

Impact: Unhandled crash of the content-analysis path on a degenerate-but-legal empty/silent transcribed chunk, aborting an otherwise-recoverable job. Reproduced: `_measure_technical_level('')` and `('   ')` raise `ZeroDivisionError`. (The finding's `'...'` example does not trigger it — `split()` returns one token.)

```python
  words = len(text.split())
  if words == 0:
    return 0.0
  score = min(1.0, term_count / (words * 0.05))
  # better: short-circuit analyze_content via MIN_WORDS_FOR_ANALYSIS for tiny samples.
```

---

**Fallback content-type detector misclassifies ordinary prose as 'dialogue' (apostrophe/colon markers)**
Location: `transcribe_pkg/core/analyzer.py:208-209,224`
Category: LogicFlow

Description: The heuristic fallback counts `:`, `"`, and `'` as dialogue markers via raw `.count()`, then classifies as dialogue when `dialogue_count > len(text.split()) / 20`. Apostrophes in contractions (it's, don't, the study's) and ordinary colons easily exceed that threshold, routing narrative text to the `dialogue_cleaning` prompt.

Impact: Misclassified content gets the dialogue cleaning strategy (preserves speaker turns / conversational tone) — wrong for narrative. This fallback fires whenever the LLM classification fails, so it compounds the `reasoning_effort="none"` bug. Reproduced: a 48-word non-dialogue paragraph → `dialogue_count=15` vs threshold 2.4 → classified 'dialogue'. Note `DIALOGUE_THRESHOLD_RATIO` is defined but unused (the `/20` is hardcoded).

```python
  dialogue_markers = ['" ', ' said ', ' asked ', ' replied ']
  dialogue_count = sum(text.count(m) for m in dialogue_markers)
  speaker_label = re.search(r'^\s*\w+\s*:', text, re.M)
  if speaker_label and dialogue_count > len(text.split()) * DIALOGUE_THRESHOLD_RATIO:
    return "dialogue"
```

---

**Failed chunk replaced with empty string still consumes a join slot, silently dropping audio with no marker**
Location: `transcribe_pkg/core/transcriber.py:292-294` (sequential) and `367-369` (parallel); retry predicate at `api_utils.py:112`
Category: ErrorHandling

Description: When a chunk transcription raises, the handler appends `""` and continues; the final text is `' '.join(transcripts)`. No error surfaced, no placeholder — a 10-minute span vanishes. Compounded by the retry decorator covering only `APIRateLimitError`/`APIConnectionError`, so a transient 500/timeout wrapped as generic `APIError` is NOT retried and goes straight to the silent-drop path.

Impact: Partial, silently-truncated transcripts presented as successes (exit 0). For paid Whisper batch work this corrupts output without notice. A per-chunk error log exists, so it is not 100% invisible, but there is no aggregate signal.

```python
  # count failures and surface them; broaden retry to transient server errors
  retry=retry_if_exception_type((APIRateLimitError, APIConnectionError, openai.APITimeoutError))
  # ...after assembly:
  if failed_chunks:
    self.logger.warning(f"{failed_chunks}/{total_chunks} chunks failed; transcript is incomplete")
    # optionally raise / return non-zero under a strict flag
```

---

**Worker exceptions silently fall back to unprocessed text, hiding systematic failures**
Location: `transcribe_pkg/core/parallel.py:121-127`; `transcribe_pkg/core/processor.py:307-309`
Category: ErrorHandling

Description: Two (actually three) layers swallow every worker exception: `parallel.py:124` substitutes the ORIGINAL chunk on `future.result()` failure; `process_chunk` (`processor.py:307`) returns the raw chunk on any failure; and `_process_chunk` does the same. No aggregate signal: if every chunk fails (bad key, 429 storm, provider error), `process_text` returns the fully-unprocessed input joined together and the CLI treats it as success.

Impact: A run where post-processing entirely failed is indistinguishable from success — the user gets the raw transcript back with grammar/hesitations intact and exit 0. Per-chunk ERROR logs exist but never reach the exit code.

```python
  failed = 0
  # ... in the except: failed += 1; chunk_results[idx] = chunks[idx]
  if failed == total_chunks:
    raise APIError(f"All {total_chunks} chunks failed post-processing; raw text would mislead")
  elif failed:
    self.logger.warning(f"{failed}/{total_chunks} chunks fell back to raw text")
```

---

**Shared CacheManager mutated concurrently from ThreadPoolExecutor workers (thread-unsafe)**
Location: `transcribe_pkg/utils/cache.py:267-300` (eviction/mutation); `processor.py:285,304,316`; `parallel.py:111-127`
Category: Concurrency *(merges 2 findings: postprocess-core + concurrency)*

Description: One shared `CacheManager` is hit by many worker threads (`use_processes=False`). `_add_to_memory` does a non-atomic check-then-act eviction: `min(self.memory_timestamp.keys(), …)` then `del` on shared dicts, with no lock and no `threading` import. `min()` over a dict another thread is mutating can raise `RuntimeError: dictionary changed size during iteration`; LRU bookkeeping updates can be lost.

Impact: Under default parallel post-processing with caching, concurrent eviction can raise mid-run; because the set() at `processor.py:316` is outside any try, the exception propagates to `future.result()` and the chunk is silently replaced by the ORIGINAL unprocessed chunk — degraded output, not a crash. The race window only opens once `memory_cache` reaches `max_memory_items` (default 1000), so it needs a long transcript with >1000 distinct keys.

```python
  import threading
  # in __init__: self._lock = threading.RLock()
  def _add_to_memory(self, hashed_key, value):
    with self._lock:
      if len(self.memory_cache) >= self.max_memory_items and self.memory_timestamp:
        oldest = min(self.memory_timestamp, key=self.memory_timestamp.get)
        self._remove_from_memory(oldest)
      self.memory_cache[hashed_key] = value
      self.memory_timestamp[hashed_key] = time.time()
  # wrap the memory branches of get()/set()/_remove_from_memory() in the same lock
```

---

**Cache key collides between specialized and standard processing of the same chunk**
Location: `transcribe_pkg/core/processor.py:284`
Category: Bug

Description: The chunk cache key derives only from `chunk, chunk_context, language, model, temperature` — not the content-analysis branch or `specialized_prompt`. The function then takes one of two materially different paths (specialized `call_llm` with `specialized_prompt`, or standard `_process_chunk` with the generic prompt), and both store under the identical key.

Impact: Cross-mode cache poisoning. If the same chunk is processed once with content analysis and once without (toggle change, or a chunk shared across two documents with different classified content types), the second run serves the first's result for the wrong prompt — subtly wrong post-processing, no error. Survives across runs (disk cache).

```python
  prompt_id = _stable(specialized_prompt) if (specialized_prompt and content_analysis) else 'standard'
  cache_key = (
    f"chunk_processing:{_stable(chunk)}:{_stable(chunk_context)}:"
    f"{language}:{self.model}:{self.temperature}:{prompt_id}"
  )
```

---

**`_get_chunk_with_complete_sentences` locates the chunk via `text.find()` (fragile to repeated/normalized content)**
Location: `transcribe_pkg/core/processor.py:540-542`
Category: Bug

Description: Companion to the Critical newline finding. Beyond the newline case, `text.find(clean_chunk)` is positionally incorrect if the stripped chunk text also appears earlier than where it was taken, and the oversize-sentence raw slice (`chunk = sentence[:max_chunk_size]`) may not be found verbatim at all (returns `-1`, yielding a wrong content-dependent offset). Verified live: a sentence spanning a newline triggers `find()==-1` and silently corrupts the boundary (`'herefor'`, duplicate `.`).

Impact: Listed separately here because the same root-cause fix (track consumption by length/offset, never re-find normalized content) resolves both this Medium framing and the Critical runaway-loop framing. See the Critical entry's fix.

```python
  # Identical fix to the Critical entry: replace text.find() with offset tracking,
  # and guard: if idx == -1: remaining_text = text[len(chunk):].lstrip()
```

---

**`AudioSegment` duration/empty checks miss zero-duration (decodable-but-silent) audio → zero chunks**
Location: `transcribe_pkg/utils/audio_utils.py:144-168` (split_audio); `107-108` (byte-only empty check)
Category: Validation

Description: `load_audio` rejects only `os.path.getsize()==0`. A file that decodes to a zero-length `AudioSegment` passes, then `range(0, 0, chunk_length_ms)` yields no iterations, so `split_audio` returns `[]`. `transcribe()` then returns `''` — a silent empty success, plus a leaked temp dir. No guard for `total_length_ms == 0`.

Impact: Empty/sub-millisecond audio yields an empty transcript reported as success (exit 0), hard to distinguish from a genuinely silent recording. Reachable via header-only/truncated-but-decodable files. Uncommon but possible.

```python
  audio = self.load_audio(audio_path)
  total_length_ms = len(audio)
  if total_length_ms == 0:
    raise ValueError(f'Audio file contains no audio data: {audio_path}')
  temp_dir = tempfile.mkdtemp()  # create only after the guard
```

---

**Subtitle splitter emits empty cues and misplaces text when a long segment has fewer words than parts**
Location: `transcribe_pkg/utils/subtitle_utils.py:86-115` (generate_srt) and `171-198` (generate_vtt)
Category: Bug *(merges 2 findings: infra + dataflow)*

Description: For segments longer than `max_duration`, `num_parts = int(dur/max_duration)+1` and `words_per_part = len(words) // num_parts`. When words are sparse (slow speech, music, long pause), `words_per_part == 0`: parts 0..n−2 get `words[0:0]` = empty, and the last part grabs ALL words. The per-segment empty-text guard is never re-checked per part, so blank cues are written and all text lands on the final (wrong-timestamp) cue.

Impact: Malformed `.srt`/`.vtt` with empty cues (rejected by some players) and badly synchronized text for any over-long, word-sparse segment. No transcript data loss (all words still appear, on the last cue). Reproduced: 2 words / 12s / max 5s → entries 1-2 blank, 'hello world' on entry 3. Applies identically to both generators.

```python
  num_parts = max(1, min(int((end_time - start_time) / max_duration) + 1, len(words)))
  words_per_part = max(1, len(words) // num_parts)
  # ... inside the loop:
  part_text = " ".join(words[start_idx:end_idx]).strip()
  if not part_text:
    continue
```

---

**Local-whisper GPU model load has no CPU fallback, breaking the documented graceful-degradation contract**
Location: `transcribe_pkg/utils/local_whisper.py:94-112`
Category: ErrorHandling

Description: `_detect_device()` returns `'cuda'` in auto mode whenever `torch.cuda.is_available()`, but `faster-whisper`/`ctranslate2` loads the GPU model via cuDNN/cuBLAS that torch availability does NOT guarantee. `_get_model()` wraps the `WhisperModel(device='cuda', ...)` construction in no `try/except`, so a raw `RuntimeError` propagates to the top-level handler (exit 1). CLAUDE.md states "Failing gracefully when GPU is absent is the contract."

Impact: On a workstation where torch sees the GPU but ctranslate2's CUDA libs are missing/mismatched (a common state), `transcribe --local` hard-fails with a cryptic library error instead of falling back to CPU int8. The only "fallback" is help text telling the user to pass `--device cpu`.

```python
  try:
    self._model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
  except Exception as e:
    if device == 'cuda' and self.device == 'auto':
      self.logger.warning(f'CUDA model load failed ({e}); falling back to CPU')
      self._model = WhisperModel(self.model_size, device='cpu', compute_type='int8')
    else:
      raise  # honour an explicit --device cuda request
  return self._model
```

---

**Free-function `transcribe_audio()` dispatches on a mutable module global and calls `.audio` on the wrong client type**
Location: `transcribe_pkg/utils/api_utils.py:495-497` (with `322-326`)
Category: LogicFlow *(merges 2 findings: llm-providers + errors-resources)*

Description: The exported wrapper guards only `if openai_client is not None:` then calls `openai_client.audio.transcriptions.create(...)`, assuming the global is a raw SDK client (the test-injected mock). But `get_openai_client()` also assigns the global to the `OpenAIClient` WRAPPER, which exposes the SDK as `.client`, not `.audio`. `call_llm`'s OpenAI-reasoning path calls `get_openai_client()`, so after any reasoning-model LLM call in the same process, a later `transcribe_audio()` call hits `AttributeError`.

Impact: Order-dependent latent failure: the public `transcribe_audio()` free function raises `AttributeError: 'OpenAIClient' object has no attribute 'audio'` once `get_openai_client()` has run. Reproduced in venv. The primary pipeline uses the wrapper's `transcribe_audio` METHOD and is unaffected — this hits direct callers/integrations.

```python
  if openai_client is not None and hasattr(openai_client, 'audio'):
    transcription = openai_client.audio.transcriptions.create(...)
    return transcription.text if hasattr(transcription, 'text') else transcription
  client = get_openai_client()
  return client.transcribe_audio(...)
```

---

**`clean-transcript` performs no numeric validation on temperature / max-tokens / max-chunk-size**
Location: `transcribe_pkg/cli/commands.py:622`
Category: Validation

Description: `transcribe_command` runs `_validate_numeric_args`, but `clean_transcript_command` accepts the same numeric options (`-t`, `-M`, `-s`) and calls no validator (the shared validator can't be reused as-is — it references `max_workers` fields the namespace lacks). Values flow unchecked into the processor.

Impact: `clean-transcript raw.txt -s 0` causes a `ZeroDivisionError` deep in the chunker (verified: `int((total_length / self.max_chunk_size) * 2)` at `processor.py:352`); `-M -1`/`-t 9` reaches the provider as an opaque 400. Crash or opaque error on clearly-invalid input; UX inconsistency with the sibling command.

```python
  if not 0.0 <= parsed_args.temperature <= 1.0:
    logger.error(f"Invalid --temperature: {parsed_args.temperature}. Must be 0.0-1.0."); return 1
  if parsed_args.max_tokens < 1:
    logger.error(f"Invalid --max-tokens: {parsed_args.max_tokens}. Must be >= 1."); return 1
  if parsed_args.max_chunk_size <= 0:
    logger.error(f"Invalid --max-chunk-size: {parsed_args.max_chunk_size}. Must be > 0."); return 1
```

---

**Empty/whitespace input still issues LLM calls instead of short-circuiting**
Location: `transcribe_pkg/core/processor.py:142-194`
Category: Validation

Description: `process()` does `text = text.rstrip()` but never returns early for empty/whitespace input. With `use_parallel=True` and `total_length <= max_chunk_size`, the small-text branch builds `chunks = [text[:split_point], text[split_point:]]` → `['','']` for empty text, and both empty chunks are dispatched to the LLM. (The sequential branch correctly short-circuits.)

Impact: Wasted paid API calls on empty audio/files, plus risk of injecting model-invented boilerplate into an otherwise-empty transcript. Low-frequency edge case, no corruption of genuine content.

```python
  text = text.rstrip()
  if not text.strip():
    self.logger.info('Empty transcript; nothing to process')
    return ''
  # and in the small-text branch: if not chunk.strip(): results.append(''); continue
```

### Low

---

**Small-text "parallel" branch is sequential and splits mid-sentence**
Location: `transcribe_pkg/core/processor.py:175-191`
Category: LogicFlow

Description: When `use_parallel=True` but `total_length <= max_chunk_size`, `process()` splits at an arbitrary 40% character offset (`split_point = max(1, int(total_length * 0.4))`) and processes the two halves in a plain sequential loop — not via the parallel processor. The split is a raw character index, routinely cutting a sentence in half; halves are joined with `\n\n`.

Impact: For small input with `--parallel`, a sentence is split across two LLM calls, producing a broken fragment and an unwanted paragraph break, for text already small enough to process in one clean call. Pure downside (no speed gain — it is sequential).

```python
  result = (self._process_with_content_analysis(text, context, language)
            if (content_analysis and self.content_aware)
            else self._process_chunk(text, context, None, language))
  # if a real 2-way split is wanted, split on a sentence boundary, not a 0.4 char index
```

---

**First chunk seam never overlaps (off-by-one in `len(chunks) > 1` guard)**
Location: `transcribe_pkg/utils/text_utils.py:202`
Category: LogicFlow

Description: The overlap back-up block is guarded by `if overlap > 0 and len(chunks) > 1:`. After the first chunk is appended, `len(chunks)==1`, so the rewind is skipped and the chunk-0/chunk-1 boundary gets zero overlap while every later boundary overlaps.

Impact: Inconsistent cross-chunk context: the first seam loses what all others get. No corruption (forward progress is independently guaranteed). Minor, but a genuine off-by-one in the overlap window. (Becomes moot if overlap is set to 0 per the Critical fix.)

```python
  if overlap > 0 and chunks and sentence_index < total_sentences:
    # drop the `len(chunks) > 1` condition
```

---

**Cross-chunk context (`prev_context`) is always None in parallel mode**
Location: `transcribe_pkg/core/processor.py:328` (also `280, 312`); `parallel.py:115`
Category: LogicFlow

Description: The sequential path maintains a rolling `context_summary` between chunks; the parallel path calls `process_text(prev_context=None)`, and `process_text` passes the same shared kwargs to every chunk, so `kwargs.get("prev_context")` is always None. Concurrent chunks cannot thread a previous-chunk summary.

Impact: Parallel output is lower-quality than sequential at chunk boundaries (pronoun/topic/terminology continuity). Invisible unless compared to sequential. Arguably an inherent trade-off of concurrency.

```python
  # Accept and document the trade-off, or bridge context via the overlap region:
  # pass the preceding chunk's tail into the system prompt as prev_context.
  # At minimum, add a comment: parallel mode trades context-continuity for speed.
```

---

**`process_text()` never injects `chunk_index` — every worker logs `chunk 0`**
Location: `transcribe_pkg/core/parallel.py:114-116`
Category: LogicFlow

Description: `process_audio_chunks()` copies kwargs per chunk and sets `chunk_kwargs['chunk_index'] = i`; `process_text()` does NOT — it submits the same shared kwargs with no index. The worker reads `kwargs.get('chunk_index', 0)`, so all chunks report index 0.

Impact: Per-chunk debug/error logs in the parallel post-processing path are all labelled `chunk 0`, making it impossible to identify which chunk failed/was cache-served. Output ordering is correct (keyed by `future_to_idx`). Log-label accuracy only.

```python
  for i, chunk in enumerate(chunks):
    chunk_kwargs = kwargs.copy()
    chunk_kwargs['chunk_index'] = i
    future = executor.submit(process_func, chunk, chunk_kwargs)
    future_to_idx[future] = i
```

---

**`_default_reasoning_effort` applies to ALL reasoning models, not the "gpt-5.4+" its docstring claims**
Location: `transcribe_pkg/utils/api_utils.py:69-77`
Category: Bug

Description: The docstring says "gpt-5.4+ reasoning models", but the implementation delegates to `_is_reasoning_model()`, which matches every `gpt-5*`, `o1*`, `o3*`. A reader trusting the docstring would assume o1/o3/gpt-5 are unaffected when they are not.

Impact: Documentation/behavior mismatch that hides the blast radius of the `reasoning_effort="none"` bug. Standalone, a docstring-wording nit.

```python
  # If returning "minimal" for all reasoning models (per the SDK fix), align the comment:
  # "Default reasoning_effort for all reasoning models (gpt-5*, o1*, o3*)."
```

---

**Production `transcribe` command bypasses the SIGINT/KeyboardInterrupt handler (lives only in main.py)**
Location: `transcribe_pkg/__main__.py:10-13`
Category: ErrorHandling

Description: Clean Ctrl-C handling (`signal.signal` + `try/except KeyboardInterrupt`) is defined only in `main.py`. The on-PATH `transcribe` wrapper runs `python -m transcribe_pkg`, routing through `__main__.py`, which imports `transcribe_command` directly and never imports `main.py`, so the handler is never installed. `KeyboardInterrupt` is not caught by the broad `except Exception` either.

Impact: Pressing Ctrl-C during a long transcription via the real `transcribe` command prints a raw traceback and exits 1, instead of clean `^C` + exit 130. Cosmetic/UX; no data loss.

```python
  # __main__.py — install the handler on the path actually used:
  from transcribe_pkg.main import transcribe_main  # importing main.py registers signal.signal
  if __name__ == "__main__":
    sys.exit(transcribe_main())
```

---

**Transcript output files written without `encoding='utf-8'`**
Location: `transcribe_pkg/cli/commands.py:212, 354, 665`; `transcribe_pkg/core/transcriber.py:542` (the one correct site)
Category: ErrorHandling

Description: Transcript writes use `open(path, 'w')` with no encoding, relying on the process locale. Transcripts are routinely non-English; under an explicitly non-UTF-8 locale `f.write()` raises `UnicodeEncodeError` mid-write after truncating the file. (`transcriber.py:542` and `subtitle_utils` correctly use UTF-8 — inconsistent.) Note: Python 3.12+ enables UTF-8 mode under plain `LANG=C` (PEP 540), so this only bites when UTF-8 mode is explicitly disabled (`PYTHONUTF8=0`) or under a named non-UTF-8 locale.

Impact: Under such a locale, transcribing non-ASCII content raises `UnicodeEncodeError` and leaves a truncated/empty output file; the user loses paid Whisper+LLM results. Uncommon configuration.

```python
  with open(output_path, "w", encoding="utf-8") as f: ...
  # apply to commands.py:212, 354, 665 and the read sites (634, 730)
```

---

**`split_audio` / `AudioProcessor` leaks the mkdtemp() directory on every run**
Location: `transcribe_pkg/utils/audio_utils.py:148` (mkdtemp) vs `58-67` (cleanup)
Category: ResourceLeak *(merges 2 findings: transcribe-core + errors-resources)*

Description: `split_audio()` creates a fresh `tempfile.mkdtemp()` per call and tracks only the chunk file paths. `cleanup()` `os.remove()`s the files but never removes the directory (no `rmtree`/`rmdir`; `shutil` not imported). `temp_dir` is a local, never stored on `self`.

Impact: Every transcription leaves one empty directory under the system temp dir. Over many batch runs (e.g. the `vrecord -t` integration) `/tmp` accumulates empty dirs. Small (inode metadata); most systems clear `/tmp` on reboot.

```python
  import shutil
  # __init__: self.temp_dirs = []
  # split_audio: self.temp_dirs.append(temp_dir)
  # cleanup():
  for d in self.temp_dirs:
    shutil.rmtree(d, ignore_errors=True)
  self.temp_dirs = []
```

---

**`_setup_cuda_paths` mutates `LD_LIBRARY_PATH` after process start (unreliable for the running linker)**
Location: `transcribe_pkg/utils/local_whisper.py:66-92` (called from `_get_model:100`)
Category: Bug

Description: `_setup_cuda_paths` sets `os.environ['LD_LIBRARY_PATH']` immediately before importing `faster_whisper`. The glibc loader reads `LD_LIBRARY_PATH` at process startup; mutating `os.environ` afterwards does not reliably change the search path for libraries `dlopen`-ed by the current process (it only propagates to children). On modern ctranslate2 wheels carrying RUNPATH, the edit is a harmless no-op; where the libs genuinely need it at dlopen time, it fails.

Impact: Best-effort CUDA path wiring may silently not take effect, sending users down the (unhandled) GPU-load-failure path even when libs are vendored in the venv.

```python
  # Load CUDA libs explicitly via ctypes before importing faster_whisper:
  import ctypes
  for so in discovered_cublas_cudnn_so_paths:
    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
  from faster_whisper import WhisperModel
  # Combined with the CPU-fallback fix above, this stops being user-visible.
```

---

**`ParallelProcessor.use_processes=True` is broken — worker is an unpicklable closure**
Location: `transcribe_pkg/core/parallel.py:32,105,181,115`; closure at `processor.py:276`
Category: Concurrency

Description: `use_processes` is a public ctor option switching to `ProcessPoolExecutor`, but the only `process_func` used is a nested closure (`processor.py:276`) capturing `self` (a `TranscriptProcessor` holding an OpenAI client) and locals. `ProcessPoolExecutor` must pickle the callable and args; a local closure / client-bearing `self` is unpicklable. The sole in-repo caller hardcodes `use_processes=False`, so it is latent.

Impact: A documented/public option fails at runtime with a PicklingError rather than degrading or being rejected. Misleading API surface; never exercised in-repo.

```python
  # KISS: the workload is I/O-bound API calls, so threads are correct. Drop the
  # ProcessPoolExecutor branch and the use_processes parameter entirely.
```

---

**`save_subtitles`: explicit `format_type` is silently overridden by the output-path extension**
Location: `transcribe_pkg/utils/subtitle_utils.py:246-249`
Category: LogicFlow

Description: The selection is `if format_type=='srt' or path.endswith('.srt')` then `elif format_type=='vtt' or path.endswith('.vtt')`. An explicit `format_type='vtt'` with a `.srt` path matches the first branch on the extension and writes SRT content. Explicit parameter should win. (Note: `save_subtitles` is NOT exported in `__all__`, and the only caller always agrees path+format, so there is currently zero reachable trigger — latent.)

Impact: A subtitle file whose content does not match its extension/the caller's request, if a future/external caller passes disagreeing arguments. No current real-world impact.

```python
  fmt = (format_type or '').lower()
  if fmt not in ('srt', 'vtt'):
    fmt = 'vtt' if output_path.lower().endswith('.vtt') else 'srt'
  content = generate_srt(segments) if fmt == 'srt' else generate_vtt(segments)
```

---

**SRT/VTT millisecond field truncates instead of rounding**
Location: `transcribe_pkg/utils/subtitle_utils.py:33 and 50`
Category: Bug

Description: `milliseconds = int((seconds - int(seconds)) * 1000)` truncates toward zero. `59.9996s` → `00:00:59,999` instead of `00:01:00,000`. Each timestamp is formatted independently, so it is a bounded per-timestamp error (no accumulating drift, despite the original "cumulative" framing).

Impact: Sub-millisecond inaccuracy and occasional off-by-1ms cue boundaries. Cosmetic for most playback; technically wrong rounding. Verified at runtime.

```python
  total_ms = int(round(seconds * 1000))
  hours, rem = divmod(total_ms, 3600_000)
  minutes, rem = divmod(rem, 60_000)
  secs, millis = divmod(rem, 1000)
  return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"  # ',' for SRT, '.' for VTT
```

---

**`process_audio_chunks` silently drops failed chunks, breaking positional correspondence**
Location: `transcribe_pkg/core/parallel.py:219-222`
Category: ErrorHandling

Description: On a chunk error, `chunk_results[idx] = None`, then assembly filters Nones: `[chunk_results[i] for i in range(total_chunks) if chunk_results[i] is not None]`. A dropped chunk shifts indices and breaks any positional/timestamp alignment — inconsistent with the text path, which substitutes the original chunk. `process_audio_chunks` has zero callers today (transcriber uses its own executor), so it is effectively dead.

Impact: Currently low (dead code). If ever wired into the transcriber, a transient audio-chunk failure yields a silently shortened transcript and broken segment alignment.

```python
  # Preserve positions; let combine_func decide or re-try:
  ordered_results = [chunk_results[i] for i in range(total_chunks)]  # may contain None
  # at minimum, log the missing segment index clearly.
```

---

**Cache atomic-write leaves orphaned `.tmp` files on crash; `clear()` only deletes `.cache`**
Location: `transcribe_pkg/utils/cache.py:160-167` and `clear()` `201-211`
Category: ResourceLeak

Description: `set()` writes to `{cache_path}.tmp` then `os.replace()`. If `pickle.dump()` or the process dies between open and replace, the `.tmp` survives: the `except` logs and swallows without unlinking, and `clear()` only removes `*.cache` (line 204). `get_stats()` also undercounts them. The bare `except:` at line 207 is the E722 ruff flag.

Impact: `~/.cache/transcribe` slowly accumulates orphaned `*.cache.tmp` that `clear()` won't reclaim. Minor — triggers only on uncommon mid-write failures.

```python
  try:
    with open(tmp_path, "wb") as f: pickle.dump(value, f)
    os.replace(tmp_path, cache_path)
  except Exception as e:
    self.logger.warning(f"Error saving cache to disk: {e}")
    try: os.remove(tmp_path)
    except OSError: pass
  # clear(): if file_name.endswith((".cache", ".tmp")):  and use `except OSError:`
```

---

**Provider-client factories swallow the ImportError cause, hiding why an SDK fails to load**
Location: `transcribe_pkg/utils/providers/registry.py:140-143, 159-162, 178-182, 193-196`
Category: ErrorHandling

Description: All four `_create_*_client` factories do `except ImportError: raise ProviderError("... SDK not installed ...")` with no `from e`. `ImportError` fires not only when the package is missing but also when an installed SDK fails to import (broken transitive dep, ABI mismatch). Discarding the cause turns "anthropic imported but its grpc dep is broken" into a misleading "Anthropic SDK not installed".

Impact: Misdiagnosis of provider load failures; reinstalling the top-level package doesn't help. (The implicit `__context__` is still shown in a full traceback, so the cause is not entirely invisible — only the user-facing message misleads.)

```python
  except ImportError as e:
    raise ProviderError(f"Anthropic SDK unavailable ({e}). Install/repair with: pip install anthropic") from e
  # apply `as e` / `from e` to all four factories
```

---

**Content-analysis threshold constants are defined and exported but never used (incl. the missing short-text guard)**
Location: `transcribe_pkg/constants.py:83-89` vs `transcribe_pkg/core/analyzer.py:224-231`
Category: LogicFlow

Description: `MIN_WORDS_FOR_ANALYSIS`, `DIALOGUE_THRESHOLD_RATIO`, `TECHNICAL_THRESHOLD_RATIO`, `SPEECH_THRESHOLD_COUNT`, `LECTURE_THRESHOLD_COUNT` are declared, documented, and `__all__`-exported, but `analyzer.py` hardcodes the equivalent magic numbers (`/20`, `/100`, `>3`) and never references them. `MIN_WORDS_FOR_ANALYSIS` is exactly the guard that would prevent the `_measure_technical_level` crash. (The inline literals currently match the constants, so no drift today — purely a maintenance hazard.)

Impact: Dead constants give a false impression of centralized, tunable thresholds; the short-text guard is absent. Maintainability only.

```python
  # gate analyze_content on MIN_WORDS_FOR_ANALYSIS and replace inline literals:
  if len(text.split()) < MIN_WORDS_FOR_ANALYSIS:
    return "general"
  # ... use DIALOGUE_THRESHOLD_RATIO / TECHNICAL_THRESHOLD_RATIO / SPEECH_/LECTURE_THRESHOLD_COUNT
```

---

**Gemini client: `response.text` raises on blocked/empty candidates rather than returning `''`**
Location: `transcribe_pkg/utils/providers/gemini_client.py:60`
Category: ErrorHandling

Description: `return response.text or ""` assumes `.text` is always accessible. In google-generativeai, the `.text` accessor raises `ValueError` when there is no valid Part (safety block, `finish_reason != STOP`, empty candidates); the `or ""` never executes because the attribute access itself throws. The other clients degrade to `""`. (SDK not installed here — from the documented accessor contract.)

Impact: Gemini calls that get safety-blocked raise an unwrapped `ValueError` out of the provider, inconsistent with the empty-string contract; callers catching `APIError`/`EmptyResponseError` won't catch it. Conditional (Gemini is non-default).

```python
  if not getattr(response, 'candidates', None):
    return ""
  try:
    return response.text or ""
  except ValueError:
    return ""
```

---

**`call_llm()` docstring says it raises `EmptyResponseError`, but the production path silently returns `''`**
Location: `transcribe_pkg/utils/api_utils.py:357-359` (docstring) vs `288/424/458-467`
Category: ErrorHandling

Description: The docstring and the test/mock branch raise `EmptyResponseError` on empty; the production branches do not (`chat_completion` logs and returns `""`; provider clients return `""`). A documented exception is never raised in production. (Impact is narrower than first reported: `analyzer.py` and `_extract_domains` already explicitly handle empty replies with heuristic fallback + debug logs, so empty responses are observable, not silent.)

Impact: Contract/docstring drift. Callers treating `call_llm` as "never silently empty" get `""` instead.

```python
  # Pick one and make all branches consistent. Simplest: align the docstring:
  #   Returns: str — model response text, or "" if the response was empty.
  # (auxiliary callers already have explicit fallbacks)
```

---

**Prefix routing: `gpt-` captures Ollama-hosted `gpt-oss`; several common local-model names unmapped**
Location: `transcribe_pkg/utils/providers/registry.py:25-36` (PREFIX_MAPPING) and `api_utils.py:439-443`
Category: LogicFlow

Description: `gpt-` routes to OpenAI, so `gpt-oss:20b` (Ollama) is misrouted. `mistral` does not match `mixtral`; `gemma`, `codellama`, `llava`, `deepseek-r1`, bare `o1`/`o3`, `o4-*` are unmapped. `call_llm` masks the unmapped case by falling back to `provider='openai'` on `ProviderError`, so a bare `o1`/`gemma` silently goes to OpenAI instead of raising the registry's helpful error.

Impact: Edge-case misrouting for Ollama users; unhelpful failure mode for common local models (must use `--provider`); a confusing OpenAI error/charge instead of explicit guidance. Workarounds exist (`ollama/` prefix, `--provider`).

```python
  # add common Ollama prefixes and special-case gpt-oss before the generic gpt- check:
  PREFIX_MAPPING.update({'gemma': 'ollama', 'gemma2': 'ollama', 'mixtral': 'ollama',
                         'codellama': 'ollama', 'llava': 'ollama', 'deepseek': 'ollama'})
  if model_lower.startswith('gpt-oss'): return 'ollama'
  # reconsider the call_llm ProviderError->openai fallback for unknown models
```

---

**Second None-content `.strip()` crash in `_call_llm_impl`**
Location: `transcribe_pkg/utils/api_utils.py:563`
Category: ErrorHandling

Description: `content = response.choices[0].message.content.strip()` assumes a str; `content=None` raises `AttributeError`, and the surrounding try catches only `openai.APIError`, so it escapes as the wrong exception type rather than `EmptyResponseError`. (`_call_llm_impl` has zero callers — not production, not tests — so it is effectively dead code.)

Impact: Were it called, a null-content model response would surface as an uncaught `AttributeError` instead of `EmptyResponseError`. No reachable impact today.

```python
  raw = response.choices[0].message.content
  if not raw or not raw.strip():
    raise EmptyResponseError("Empty content in API response")
  return raw.strip()
```

---

**`language_codes_command` omitted from `cli/__init__.py` `__all__`**
Location: `transcribe_pkg/cli/__init__.py:8-12`
Category: Other

Description: `__all__` lists three commands but omits `language_codes_command`, a fourth public command (wired in `setup.py` console_scripts, imported by `main.py`). `__init__.py` has no imports, so `__all__` is purely declarative; `main.py` imports by explicit path, so there is zero runtime effect.

Impact: Cosmetic/maintenance — the export list misrepresents the public CLI surface.

```python
  __all__ = [
    "transcribe_command",
    "clean_transcript_command",
    "create_sentences_command",
    "language_codes_command",
  ]
```

---

**Dead infra: `progress.ParallelProcessor` and the `cached()` decorator are never used**
Location: `transcribe_pkg/utils/progress.py:171-257` (ParallelProcessor); `transcribe_pkg/utils/cache.py:316-363` (cached)
Category: Architecture

Description: `progress.py` defines a `ParallelProcessor` that nothing imports (production uses the one in `core/parallel.py`) — two same-named classes with different APIs. `cache.py`'s `cached` decorator is imported into `processor.py` but never applied (`@cached` has zero matches); it builds keys from `str(args)`+`sorted(kwargs)` (same cross-process instability as the hash() finding) and instantiates a module-shared `CacheManager` at decoration time. Not auto-flagged by ruff (module-level defs).

Impact: Confusing duplicate API surface; a future contributor may wire up the wrong `ParallelProcessor` or the broken-keyed `cached()`, reintroducing the disk-cache-miss bug. No current runtime effect.

```python
  # Delete progress.ParallelProcessor (keep ProgressDisplay) or rename it
  # (e.g. SimpleParallelProcessor). Remove the unused `cached` import from
  # processor.py:26 and either delete cached() or fix its key construction.
```

---

## Flow-of-Logic Issues

The user asked for special attention to control-/data-flow defects. Re-listed here by title + location (severity in brackets). These are the cases where the program follows a wrong path or threads data incorrectly — distinct from crashes or pure validation gaps.

- **[Critical] Newline corrupts text + runaway loop via `text.find()` on normalized content** — `processor.py:540-544`. Data-flow: the located span no longer exists in the source after newline→space normalization; `find()` returns −1 and the loop re-consumes garbage.
- **[Critical] Chunk overlap added but never removed at reassembly** — `parallel.py:145`, `processor.py:117,325`. Data-flow: overlap region intended only as LLM context is emitted into the output; the dedup routine (`adjust_chunk_boundaries`) is unreachable.
- **[High] Cross-chunk timestamp offset uses last-segment end, not chunk duration** — `transcriber.py:273-279, 450-456`. Data-flow: per-chunk-relative timestamps stitched with the wrong accumulator → cumulative drift.
- **[High] Skipped/failed chunk does not advance timestamp offset** — `transcriber.py:378-456, 196-279`. Control-flow: `continue` bypasses the offset update guarded inside the success branch → full-chunk-width jump.
- **[High] Sequential reassembly glues words when a chunk lacks terminal punctuation** — `processor.py:406-408`. Control-flow: the space-insertion branch is gated on a too-narrow punctuation whitelist.
- **[Medium] Anthropic client drops `temperature=0.0`** — `anthropic_client.py:53-54`. Control-flow: `if temperature > 0` skips the legitimate deterministic 0.0 case.
- **[Medium] Specialized-prompt templates referenced but undefined; fallback silently uses generic** — `analyzer.py:103-129`. Control-flow: the content-aware branch always degrades to the generic template.
- **[Medium] Config precedence inverted (env overrides JSON file)** — `config.py:95-99`. Control-flow: file applied before env, env overwrites unconditionally.
- **[Medium] Fallback detector misclassifies prose as dialogue** — `analyzer.py:208-209,224`. Data-flow: contraction apostrophes counted as dialogue markers route text to the wrong cleaning strategy.
- **[Medium] Cache key collides between specialized and standard processing** — `processor.py:284`. Data-flow: the discriminating prompt identity is absent from the key.
- **[Medium] Free-function `transcribe_audio()` dispatches on a mutable global, calls `.audio` on the wrong client shape** — `api_utils.py:495-497`. Control-flow: an order-dependent global mutation selects the wrong attribute path.
- **[Low] Small-text "parallel" branch is sequential and splits mid-sentence** — `processor.py:175-191`.
- **[Low] First chunk seam never overlaps (off-by-one)** — `text_utils.py:202`.
- **[Low] `prev_context` always None in parallel mode** — `processor.py:328`.
- **[Low] `process_text()` never injects `chunk_index` (all logs say chunk 0)** — `parallel.py:114-116`.
- **[Low] Prefix routing misroutes `gpt-oss`; `call_llm` masks unmapped models by falling back to OpenAI** — `registry.py:25-36`, `api_utils.py:439-443`.
- **[Low] `process_audio_chunks` drops failed chunks, breaking positional correspondence** — `parallel.py:219-222` (dead code).

---

## Test Coverage Notes

The suite is substantial (3,374 test lines / 16 files) but the verifications surfaced **specific gaps where mocking hides real defects**:

- **`transcribe_audio_file()` TypeError is masked** by `tests/test_integration.py:33` patching out the entire `Transcriber` class, so the real `__init__` (which rejects `max_workers`) never runs. A non-class-mocked smoke test would catch this guaranteed crash. *(High finding.)*
- **The disk-cache cross-run miss is invisible to tests** because the in-process memory cache works (builtin `hash()` is stable within one process). A test that hashes in a subprocess, or asserts a stable on-disk filename across two `CacheManager` instances with fresh interpreters, would expose it. *(High finding.)*
- **Content-aware specialization is under-tested per the project's own convention.** CLAUDE.md mandates "add both the prompt and a test case" for content types, yet `speech`/`lecture` cleaning and all `*_summary` templates are referenced-but-undefined and silently fall back — a test asserting that a 'speech'-classified sample receives a speech-specific prompt would have caught the dead mapping. *(Medium finding.)*
- **Free-function `transcribe_audio()` is exercised** (`tests/test_audio_integration.py:48`) but only in a state where the module global is still a mock or None; no test runs `get_openai_client()` first, so the `.audio`-on-wrapper crash is not covered. *(Medium finding.)*

Recommendation: add at least one **non-mocked-class** construction test per public entry point (`transcribe_audio_file`, `transcribe_audio`), a subprocess-based cache-key stability test, and a content-type→prompt-name assertion per declared content type.

---

## Dependency Notes

- **Installed `openai==2.6.1`** defines `ReasoningEffort = Optional[Literal["minimal","low","medium","high"]]`. The uncommitted edit emitting `reasoning_effort="none"` (and docstrings advertising `"none"`/`"xhigh"`) is **ahead of the installed SDK** and will be rejected with a 400 for OpenAI reasoning models. Either pin/upgrade the SDK to a version that supports the new enum *and* gate the value behind a version check, or revert to `"minimal"`. The mypy OpenAI signature errors (`messages` param type, `transcribe`/`create` overloads) point at the same drift between hand-built dicts and current SDK types.
- **`faster-whisper` / `ctranslate2` (local path)** pin only `faster-whisper>=1.0.0` (`requirements-local.txt`); the CUDA-library coupling is handled by a fragile post-start `LD_LIBRARY_PATH` edit with no CPU fallback. This is a runtime-environment dependency hazard rather than a version pin issue — see the two `local_whisper.py` findings.
- **`google-generativeai`** is not installed in the audit environment; the Gemini `.text` finding rests on the documented accessor contract, not a live repro. If Gemini is a supported provider, add it to the dev/test extras so the empty-response path can be exercised.
- No `pyproject.toml`: ruff/mypy run on defaults. Adding one with `[tool.ruff]` (2-space indent already in use) and `[tool.mypy] strict` for new modules would lock in lint/type expectations and surface the 54 mypy errors in CI.

#fin
