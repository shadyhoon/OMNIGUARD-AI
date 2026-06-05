# 🛡️ OmniGuard AI

> **A real-time multimodal verification layer for AI-generated content.**
> Built with Streamlit, FastAPI, OpenCV, ChromaDB, DuckDuckGo, and OpenAI.

OmniGuard AI is a unified trust layer that scores the veracity of a news article, video link, image, audio clip, or social-media thread in real time. It combines a deterministic forensic pipeline, a Retrieval-Augmented Generation (RAG) cross-reference engine, an OpenCV vision layer, and an optional OpenAI augmentation path.

![OmniGuard AI hero](docs/screenshot-placeholder.png)
<sub>_Add a real screenshot of the dashboard here before publishing._</sub>

---

## ✨ Features

- **Three content-type toggles, one unified button** — 🎬 Video Link · 📰 Text Article · 🧵 Social Media Thread.
- **Forensic Veracity Score (0-100)** rendered as a Plotly donut chart, colour-coded emerald / amber / red.
- **OpenCV vision layer** for image and video frames — colour-cast / lighting consistency, JPEG blockiness, inter-frame luminance variance, edge density.
- **RAG cross-reference engine** — sentence-transformers embeddings + local ChromaDB vector store of trusted truths, with a DuckDuckGo live-web fallback for unknown claims.
- **Hallucination & Forensic Warnings** that flag generative-AI tell-tales, long sentence patterns, low lexical diversity, and assertive absolute language.
- **Optional LLM Augmentation** — when `OPENAI_API_KEY` is set as a system environment variable, an extra panel adds per-claim neutral summaries and creative-leap detection. **🔬 LLM call evidence** expander shows the model name, response id, latency, and tokens for every call — proof the call actually happened.
- **yt-dlp video downloader** for the Video Link toggle — caps file size and length so it stays polite to your disk and your bandwidth.
- **FastAPI backend** at `api.py` exposing `/`, `/health`, `/analyze`, `/analyze/upload`. The Streamlit dashboard can call it directly (toggle in **⚙️ Advanced options**) or run the pipeline in-process.
- **Dark, ultra-modern UI** with a glassy card layout, gradient hero, pill-style radio toggle, and status badges throughout.

---

## 🚀 Quick start

### 1. Clone & enter the project
```bash
git clone <your-repo-url> omniguard-ai
cd omniguard-ai
```

### 2. Create the virtual environment and install dependencies

**Lightweight install** (heuristics + LLM only, ~150 MB):
```bash
pip install -r requirements.txt
```

**Full-stack install** (RAG, OpenCV, FastAPI, yt-dlp, ~3 GB):
```bash
pip install -r requirements-full.txt
```

The full stack is recommended — the heavy dependencies (OpenCV, Chroma, sentence-transformers, yt-dlp) are imported lazily, so the dashboard still boots in a few seconds even when they're installed.

**Windows convenience script:**
```powershell
.\setup_env.bat
```

**Linux / macOS:**
```bash
bash setup_env.sh
```

### 3. Run the dashboard

```bash
# Windows
.\.venv\Scripts\streamlit.exe run app.py

# Linux / macOS
.venv/bin/streamlit run app.py
```

Open <http://localhost:8501> in your browser. **No API key is required** — the heuristic pipeline works on its own.

### 4. (Optional) Run the FastAPI backend

```bash
.\.venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000
```

The interactive API docs are at <http://localhost:8000/docs>. Then in the dashboard, open **⚙️ Advanced options → Call FastAPI backend** and point it at the same URL.

---

## 🔐 Optional: enable LLM augmentation

The OpenAI key is **never** read from a project file. Set it as a **system environment variable**:

**Windows (PowerShell, persistent):**
```powershell
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-your-key", "User")
```
Then restart your terminal / IDE.

**Linux / macOS (Bash):**
```bash
echo 'export OPENAI_API_KEY="sk-your-key"' >> ~/.bashrc
source ~/.bashrc
```

The dashboard's sidebar shows a **live health probe** so you always know what's happening:

| Sidebar status            | Meaning                                                        |
|---------------------------|----------------------------------------------------------------|
| 🟢 **online**             | Probe round-trip succeeded — augmentation will run             |
| 🔴 **unreachable**        | Key is set but OpenAI is rejecting requests (e.g. no credits)  |
| ⚪ **offline**            | No key configured — heuristic-only mode                        |

A **🔄 Re-probe LLM** button is provided with a 60-second cache so the dashboard never hammers the API.

---

## 🧠 How it works

OmniGuard runs **four complementary pipelines**, all wired together through a single report dict.

### 1. Heuristic forensic pipeline — [`utils/verifier.py`](utils/verifier.py)

| Content type | What it analyses |
|--------------|------------------|
| **Text**     | Sentence / word counts, type-token ratio, punctuation & number density, LLM tell-tale phrases, assertive-language detection |
| **Image**    | Magic-byte detection, per-block Shannon entropy, block-variance and alignment-anomaly signals |
| **Video**    | Container-header parsing (MP4 / MKV / AVI / MOV), byte-level entropy proxy, frame sampling for OpenCV pass |
| **Audio**    | RIFF/WAVE header parsing, byte-level entropy, voice-consistency proxies |
| **URL**      | Fetches the first 1 MB, extracts `<article>` → `<main>` → highest `<p>`-density block via BeautifulSoup, matches against a built-in fact table with word-boundary regex, ranks by **relevance** |

### 2. RAG cross-reference engine — [`utils/rag.py`](utils/rag.py)

1. **Claim extraction** — split text into assertive claim-sized sentences.
2. **Embed** with `sentence-transformers/all-MiniLM-L6-v2`.
3. **Query ChromaDB** (local file-based vector store, seeded with 10 canonical truths on first run). Cosine threshold: 0.65.
4. **Fallback to DuckDuckGo HTML** for claims the local store has no answer to.
5. Each hit carries `engine` (`chroma` or `duckduckgo`) and `relevance`.

### 3. OpenCV vision layer — [`utils/vision.py`](utils/vision.py)

For image and video inputs:
- **Lighting consistency** — LAB colour-cast comparison between image centre and border.
- **JPEG blockiness** — 8-pixel boundary discontinuity analysis.
- **Edge density** — Canny edge fraction (over-smoothed = suspicious).
- **Inter-frame variance** for video — rapid luminance changes can indicate splicing.

### 4. Optional LLM augmentation — [`utils/llm.py`](utils/llm.py)

- Reads the key with `os.getenv("OPENAI_API_KEY")` at every call — never from any project file.
- Adds per-claim neutral summaries and creative-leap detection.
- **Every call is evidence-tracked** — model name, response id, latency, tokens in/out. Surfaced in the **🔬 LLM call evidence** expander.
- A 60-second cached health probe distinguishes the three states shown above.

### 5. FastAPI backend — [`api.py`](api.py)

| Method | Endpoint          | Purpose                                                |
|--------|-------------------|--------------------------------------------------------|
| GET    | `/`               | Tiny landing page                                      |
| GET    | `/health`         | JSON probe (heuristics, RAG, vision, LLM)              |
| POST   | `/analyze`        | Text / URL analyze, with `use_rag` and `enrich_with_llm` flags |
| POST   | `/analyze/upload` | Multipart upload for image / video / audio             |
| GET    | `/docs`           | Auto-generated Swagger UI                              |

---

## 🏗️ Project structure

```
OmniGuard AI/
├── app.py                       # Streamlit dashboard (UI + orchestration)
├── api.py                       # FastAPI backend exposing /analyze
├── components/                  # Reserved for reusable UI components
├── utils/
│   ├── verifier.py              # Heuristic analysis pipeline
│   ├── llm.py                   # Optional OpenAI augmentation + health probe
│   ├── rag.py                   # RAG cross-reference (Chroma + DuckDuckGo)
│   ├── vision.py                # OpenCV forensic vision layer
│   └── video.py                 # yt-dlp video downloader
├── data/
│   └── chroma/                  # Local ChromaDB vector store (auto-created)
├── test_app_e2e.py              # End-to-end AppTest suite
├── test_relevance.py            # Cross-reference regression tests
├── test_no_fabrication.py       # "man went to the moon" regression test
├── test_full_stack.py           # RAG + vision + API + latency suite
├── smoke_rag.py                 # One-shot RAG health probe
├── smoke_vision.py              # One-shot OpenCV health probe
├── smoke_api.py                 # One-shot FastAPI smoke test
├── requirements.txt             # Lightweight install
├── requirements-full.txt        # Full-stack install
├── setup_env.bat / .sh          # venv + dependency installer
├── .env.example                 # Template (no real keys, ever)
└── README.md                    # ← you are here
```

---

## 🧪 Testing

The project ships with **four** test suites:

```bash
# 1. End-to-end UI test - drives the real app via AppTest
.\.venv\Scripts\python.exe test_app_e2e.py

# 2. Cross-reference regression suite (word boundaries, relevance)
.\.venv\Scripts\python.exe test_relevance.py

# 3. No-fabrication regression test
.\.venv\Scripts\python.exe test_no_fabrication.py

# 4. Full-stack suite (RAG, OpenCV, FastAPI, latency)
.\.venv\Scripts\python.exe test_full_stack.py
```

Plus one-shot smoke probes for quick debugging:

```bash
.\.venv\Scripts\python.exe smoke_rag.py      # RAG end-to-end
.\.venv\Scripts\python.exe smoke_vision.py   # OpenCV forensic pass
.\.venv\Scripts\python.exe smoke_api.py      # FastAPI endpoints
```

All should print `All ... tests passed.`

---

## 🛠️ Tech stack

| Layer        | Tech                                                                                  |
|--------------|---------------------------------------------------------------------------------------|
| Frontend     | [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/python/)               |
| Backend      | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/)         |
| RAG          | [ChromaDB](https://www.trychroma.com/), [sentence-transformers](https://www.sbert.net/), [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) |
| Vision       | [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/)                             |
| Video        | [yt-dlp](https://github.com/yt-dlp/yt-dlp)                                            |
| Web fetching | [Requests](https://requests.readthedocs.io/), [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) |
| LLM          | [OpenAI Python SDK](https://github.com/openai/openai-python) (GPT-4o-mini by default)  |
| Testing      | [AppTest](https://docs.streamlit.io/library/api-reference/app-testing), [pytest]-style |
| Config       | [python-dotenv](https://pypi.org/project/python-dotenv/)                              |

---

## 📋 Design constraints

- **No secrets in the repo** — `.env` is in `.gitignore`; `.env.example` is the only template file. The OpenAI SDK is imported lazily and reads the key from `os.getenv` at call time.
- **Lazy heavy imports** — OpenCV, Chroma, sentence-transformers, yt-dlp, and torch are imported inside the functions that need them. The dashboard still boots in seconds even with the full stack installed.
- **Heuristic-first** — every LLM call has a silent-failure guard. The dashboard always renders, with or without a working API.
- **Explainable heuristics** — every score is built from inspectable measurements (TTR, entropy, magic bytes, container format, colour-cast diff, etc.), not black-box predictions.
- **Bounded downloads** — yt-dlp caps each video at 80 MB / 60 seconds so the local copy never blows up disk space.

---

## 🧩 Extending the pipeline

Adding a new verification module is straightforward:

1. Drop a new function in the appropriate `utils/*.py` module that returns a dict shaped like the existing per-modality dicts.
2. If it needs heavy deps, import them lazily and gate on `is_x_available()`.
3. Add a regression test in a new `test_*.py` file or extend `test_full_stack.py`.
4. Wire it into `app.py` and `api.py` symmetrically.

---

## 🤝 Contributing

PRs welcome. Please:
- Run all four test suites before opening a PR.
- Avoid adding hidden dependencies — list them in `requirements-full.txt` and import them lazily.
- Never commit `.env` or any real API key.

---

## 📄 License

MIT — see [`LICENSE`](LICENSE) (add this file before publishing if you want MIT).

---

<sub>Built with ❤️ for journalists, researchers, and anyone who wants a second opinion on what they read online.</sub>
