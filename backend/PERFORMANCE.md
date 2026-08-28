# STT performance architecture

How the transcription pipeline schedules CPU-bound Whisper inference so a
final transcript — the only thing on the path to an answer — never waits
behind display-only work.

## Execution model

One `FasterWhisperEngine` (one `WhisperModel`, `@lru_cache`d in
[`app/core/deps.py`](app/core/deps.py)) is shared by every capture channel.
CTranslate2 internally serialises concurrent `transcribe()` calls across
`num_workers` slots — so two independent `ThreadPoolExecutor`s, one per
channel, do not actually run in parallel; they queue inside the C++ layer in
arrival order, where nothing on the Python side can reorder them.

[`app/stt/scheduler.py`](app/stt/scheduler.py) replaces that with a single
process-wide `InferenceScheduler`: a `queue.PriorityQueue` drained by
`STT_INFERENCE_CONCURRENCY` daemon threads (default 1, matched to
`WhisperModel(num_workers=...)`). Every channel's `TranscriptionWorker`
submits into this one queue instead of owning its own executor.

```
MIC worker    ─┐                       ┌─ stt-infer-0 ─┐
               ├─ InferenceScheduler ──┤                ├─ FasterWhisperEngine (1 model)
LOOPBACK worker┘   (priority queue)    └─ stt-infer-N ─┘
```

Audio capture, VAD, and segmentation are unchanged — still one thread per
channel, still CPU-cheap enough to never fall behind real time even while
inference is busy (`test_slow_stt_does_not_block_audio_frame_consumption`).

## Job priority model

Priority is `(band, submission order)` — strict band ordering, FIFO within a
band. Default bands (lower runs first), each an env var:

| Band | Setting | Default |
|---|---|---|
| Loopback final | `STT_PRIORITY_LOOPBACK_FINAL` | 0 |
| Loopback partial | `STT_PRIORITY_LOOPBACK_PARTIAL` | 1 |
| Mic final | `STT_PRIORITY_MIC_FINAL` | 2 |
| Mic partial | `STT_PRIORITY_MIC_PARTIAL` | 3 |

A model warmup pass is submitted below all of these (see below), since it is
a prerequisite for every real job.

Consequences:

- A loopback final always overtakes any queued partial, on either channel.
- A mic backlog (candidate talking at length) cannot delay a loopback final —
  the final simply moves to the front of the shared queue.
- Rebalancing for a different product shape (e.g. mic-drives-detection) is a
  config change, not a code change.

## Partial transcript behavior

Partials are best-effort UI feedback; they must never cost the final
transcript CPU time it needs. Policy, all in
[`app/stt/pipeline.py`](app/stt/pipeline.py):

- **Coalesced**: at most one in-flight and one newest-pending partial per
  channel (unchanged from before this work).
- **Floor** (`STT_PARTIAL_MIN_AUDIO_MS`, default 500ms): a snapshot shorter
  than this is skipped — nothing useful to show yet.
- **Capped** (`STT_MAX_PARTIALS_PER_UTTERANCE`, default 4): bounds
  re-transcription of one growing utterance regardless of its length or the
  configured cadence.
- **Adaptive cadence**: the next partial fires no sooner than
  `max(STT_PARTIAL_INTERVAL_MS, time the last partial actually took)` — a slow
  machine backs off on its own instead of piling up a backlog.
- **Cancelled on speech end**: when an utterance's final is scheduled, any
  partial for that utterance still sitting in the queue (not yet running) is
  cancelled outright. A partial already running is left to finish — CTranslate2
  inference isn't interruptible — but it can never publish once a newer final
  has landed (`_published_final_utterance_id` guard, unchanged).
- **Disable switch** (`STT_ENABLE_PARTIALS=false`): for a CPU too weak to
  afford them at all; finals still run normally.
- **Backlog warning**: if the shared queue ever exceeds 6 pending jobs, one
  `inference_backlog` warning is logged suggesting a smaller model or
  disabling partials — the queue itself stays unbounded in size (each channel
  can only ever contribute two jobs) but a persistent backlog means the model
  is too slow for the machine.

## Final transcript behavior

Unchanged in spirit, strengthened in practice: a final is always submitted to
the shared scheduler at its channel's final priority, so it queues ahead of
every partial already there. On worker shutdown, a still-queued final is
*waited for*, never dropped — it's the transcript of something someone
actually said.

## Multi-channel scheduling

There is no separate "loopback lane" and "mic lane" — one queue, one set of
worker threads, ordered by the priority table above. This is deliberately
simpler than per-channel executors plus a fairness scheme: the product's
whole premise (interviewer on loopback, candidate on mic) is already encoded
as a priority order, not as topology.

## Configuration

New in this change, all in [`app/core/config.py`](app/core/config.py) /
[`.env.example`](.env.example):

| Setting | Default | Effect |
|---|---|---|
| `STT_INFERENCE_CONCURRENCY` | 1 | Scheduler threads, and `WhisperModel(num_workers=)`. Keep these coupled — raising one without the other just moves the queue into C++. |
| `STT_CPU_THREADS` | 0 | `WhisperModel(cpu_threads=)`; 0 lets CTranslate2 size its own intra-op pool. |
| `STT_ENABLE_PARTIALS` | true | Kill switch for interim transcripts. |
| `STT_PARTIAL_MIN_AUDIO_MS` | 500 | Floor below which a partial is skipped. |
| `STT_MAX_PARTIALS_PER_UTTERANCE` | 4 | Cap on re-transcription per utterance. |
| `STT_PRIORITY_LOOPBACK_FINAL/PARTIAL`, `STT_PRIORITY_MIC_FINAL/PARTIAL` | 0/1/2/3 | Queue priority bands. |

Existing `STT_MODEL`, `STT_DEVICE`, `STT_COMPUTE_TYPE`, `STT_BEAM_SIZE`,
`STT_PARTIAL_INTERVAL_MS` are unchanged. `PUT /settings` changing `stt_model`
or `stt_device` now also resets the shared scheduler singleton
(`reset_shared_scheduler()`), since its thread count derives from STT
settings.

CUDA remains explicit-opt-in only (`STT_DEVICE=cuda`); `auto` still resolves
to CPU. Not touched by this work.

## Reading the latency logs

New structured events, all `metric <event> k=v ...` at INFO via
[`app/core/metrics.py`](app/core/metrics.py) — `grep "metric "` on the log to
reconstruct one utterance's path end to end:

```
metric speech_start_detected channel=LOOPBACK utterance_id=3
metric speech_end_detected channel=LOOPBACK utterance_id=3 audio_duration_ms=2400
metric final_transcription_started channel=LOOPBACK utterance_id=3 audio_duration_ms=2400 queue_wait_ms=4
metric final_transcription_completed channel=LOOPBACK utterance_id=3 audio_duration_ms=2400 duration_ms=180 speech_end_to_transcript_ms=184 chars=42
metric question_detected session_id=... question_id=7 category=TECHNICAL_KNOWLEDGE confidence=0.91
metric question_processing_started session_id=... question_id=7
metric llm_request_started session_id=... question_id=7
metric llm_first_token_received session_id=... question_id=7 duration_ms=310
metric llm_response_completed session_id=... question_id=7 duration_ms=1150 chars=480
```

The number that matters most is `speech_end_to_transcript_ms` on the final —
it is `queue_wait_ms + duration_ms` and is the number the priority scheduler
is directly fighting to shrink. `llm_first_token_received.duration_ms` is the
Gemini-side equivalent for time-to-first-token.

`transcription_worker_metrics` (unchanged, logged on stop) still reports
per-worker totals: `partials_scheduled`, `partials_coalesced`,
`partials_skipped`, `partials_cancelled`, `errors`.

## Recommended configurations

**Low-end CPU** (2–4 cores, laptop-class):
```
STT_MODEL=distil-small.en
STT_DEVICE=cpu
STT_COMPUTE_TYPE=int8
STT_INFERENCE_CONCURRENCY=1
STT_PARTIAL_INTERVAL_MS=1500
STT_MAX_PARTIALS_PER_UTTERANCE=2
```
If `inference_backlog` warnings appear regularly, set `STT_ENABLE_PARTIALS=false`
and/or drop to `base.en`.

**Mid-range CPU** (6+ cores, desktop-class):
```
STT_MODEL=distil-small.en   # small.en for better accuracy if latency allows
STT_DEVICE=cpu
STT_COMPUTE_TYPE=int8
STT_INFERENCE_CONCURRENCY=1
STT_PARTIAL_INTERVAL_MS=1000
```
Concurrency stays at 1 even here: two channels rarely produce simultaneous
CPU-bound work in practice (the candidate and interviewer aren't usually
talking over each other), and the priority queue already means a final never
waits behind a partial regardless.

**NVIDIA GPU**:
```
STT_MODEL=distil-medium.en   # or small.en / medium.en — headroom to spend
STT_DEVICE=cuda
STT_COMPUTE_TYPE=float16
STT_INFERENCE_CONCURRENCY=1
```
CUDA stays explicit-opt-in — `STT_DEVICE=auto` never selects it, matching the
existing safety behavior in `FasterWhisperEngine._resolve_device`. This
project does not ship multiple bundled models; the model stays whatever
`STT_MODEL` names, downloaded on first use.

## Question stabilization

A question that ends mid-clause ("Can you explain what happens when") is
detected structurally (`app/realtime/question_detector.py::_looks_incomplete`
checks the last word against a fixed set of dangling conjunctions/
prepositions/articles) and is *not* asked immediately. `LiveSession` holds it
for `QUESTION_STABILIZATION_MS` (default 400ms) via a cancellable task
(`_pending_ask`); if the interviewer's continuation arrives within the
existing correction-coalesce window, the detector merges it into one complete
question and the stale timer is cancelled before it ever fires. A confident,
complete question (the overwhelming majority) is never delayed — this only
applies to the specific dangling-word case. Short follow-ups ("Why?") are
exempt regardless, since a bare interrogative would otherwise falsely trip
the same word check.

This does not (and cannot, given VAD's own ~700ms silence-close time) catch
every possible pause length — a long deliberate pause will still fire on the
incomplete fragment, same as before this feature existed. It only removes a
real number of previously-guaranteed wasted Gemini calls: any continuation
that arrives within roughly one correction-coalesce window.

## Answer modes

`app/llm/prompts.py` now routes the JSON schema hint by the *existing*
classifier category (no new classification step): CODING keeps its existing
approach/code/complexity/edge_cases shape; SQL, DEBUGGING, SYSTEM_DESIGN /
ARCHITECTURE, and BEHAVIORAL get a shared `sections: [{heading, content}]`
shape (Likely Cause/Diagnosis/Fix/Why It Works; Situation/Task/Action/Result;
etc.) — one generic field covers all of them rather than a bespoke field per
category. Everything else keeps the original summary/key_points/
detailed_answer shape. The system instruction also now explicitly states that
only the question under "CURRENT INTERVIEWER QUESTION" is to be answered;
earlier conversation is background only.

## Benchmarking

`scripts/benchmark_pipeline.py` measures six representative scenarios (short
question, multi-sentence, coding, follow-up, correction, setup+question)
through the real detection/retrieval/prompt-construction pipeline with a
zero-delay fake LLM — this isolates **application-controlled overhead only**
and is not a Gemini latency estimate. Pass `--live` with `GEMINI_API_KEY` set
to additionally run the existing `GeminiClient.benchmark_stream_latency()`
against the real API for an app-prompt-vs-minimal-prompt comparison — the only
number either benchmark produces that reflects real network/model latency.

```bash
.venv/Scripts/python.exe scripts/benchmark_pipeline.py
```

## Performance budgets

Local, deterministic stages have a budget because they are ours to control;
Gemini's own latency is reported as a distribution, not budgeted, because it
is not:

| Stage | Budget | Measured via |
|---|---|---|
| Question detection (`stt_final_to_question_detected_ms`) | < 10ms | `question_latency_trace` |
| Prompt construction (`prompt_build_ms`) | < 5ms | `question_latency_trace` |
| Retrieval, non-RAG routes | ~0ms (skipped entirely) | `question_latency_trace` |
| Retrieval, RAG/FOLLOW_UP routes | budgeted per `RAG_TOP_K`/`RAG_OVERFETCH`, not fixed | `question_latency_trace` |
| STT queue wait for a final | near-zero when a worker is free (priority scheduler) | `stt_queue_wait_ms` |
| Cancel-wait when superseding an answer | sub-millisecond (measured; not a bottleneck) | `previous_answer_cancel_wait_ms` |
| Gemini request → first text token | **not budgeted** — report p50/p95/worst observed from production logs | `gemini_request_to_first_text_token_ms` |
| End-to-end, speech-end → first visible token | as low as the above stages allow; dominated by Gemini | `total_question_to_first_visible_token_ms` |

No hardcoded SLA assertion exists for the Gemini-dependent rows — a slow
network or a loaded model would make such a test flaky for reasons outside
this codebase's control. `grep "metric question_latency_trace"` on a real log
gives the actual distribution.
