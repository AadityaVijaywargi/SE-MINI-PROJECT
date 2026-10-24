# Response-time targets (SM-68, resolves TBD-2 / PERF-7)

Targets for a single user, and what was measured on 2026-10-24 with
`tools/measure_times.py`.

**Machine:** Windows 11 laptop, Python 3.10, Flask's threaded development server, CPU only.
**Method:** 5 calls per file size through the HTTP API. Each cell shows the median (and
maximum) of two runs, written as a range when the two runs differ. The hosted instance
(Render free tier, 512 MB) still has to be measured; see SM-98 in `docs/demo-script.md`.

| Operation | Target | 200 rows (9 KB) | 10,000 rows (0.5 MB) | 50,000 rows (2.4 MB) |
|---|---|---|---|---|
| `POST /api/upload` (store, parse, profile, suggest, preview) | under 2 s up to 10 MB | 21 to 36 ms (max 53) | 26 to 57 ms (max 89) | 91 to 132 ms (max 144) |
| `POST /api/configure` (task analysis) | under 500 ms | 10 to 23 ms (max 36) | 17 to 40 ms (max 43) | 50 to 51 ms (max 58) |

| Operation | Target | Measured |
|---|---|---|
| `POST /api/train` returns 202 (PERF-1) | under 200 ms at any size | 15 ms |
| `GET /api/status` while a job trains (PERF-2) | under 200 ms | median 4 to 18 ms, max 187 ms (55 to 62 polls) |
| `GET /api/results` | under 500 ms | 4 to 15 ms |
| Default QML model: 2,000 rows, 4 qubits, 20 epochs (PERF-4) | 1 to 3 min | 57 s |
| All 8 models on the banking sample | under 2 min | 16 s |
| Container healthy after start (PERF-6) | under 40 s | checked by the CI Docker job |

Per-model training time on 10,000 rows (8,000 training rows): Random Forest 0.8 s, SVM 12.7
to 13.3 s, DNN 5.9 to 6.1 s, VQC 5.2 to 8.6 s with 3 epochs. SVM's training time grows with
roughly the square of the row count, which is why it is subsampled above 20,000 rows
(REQ-TR-2).

## How to re-measure

```bash
python tools/measure_times.py
```

```bash
python tools/measure_times.py https://<your-app>.onrender.com
```

On the hosted instance, open the app 5 minutes beforehand so the cold start (ASM-2) does
not count.
