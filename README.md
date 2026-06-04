# 🛡️ OmniGuard AI

> A real-time multimodal verification dashboard for spotting misinformation, deepfakes, and AI-generated content.

OmniGuard AI is a lightweight Streamlit dashboard that helps you decide whether a piece of content — a video link, a news article, or a social-media thread — is likely trustworthy, AI-generated, or factually dubious. It runs on modest hardware (8 GB RAM, no GPU) and avoids heavy deep-learning dependencies by combining a deterministic heuristic forensic pipeline with optional LLM augmentation.

![OmniGuard AI hero](docs/screenshot-placeholder.png)
<sub>_Add a real screenshot of the dashboard here before publishing._</sub>

---

## ✨ Features

- **Three content-type toggles, one unified button** — 🎬 Video Link · 📰 Text Article · 🧵 Social Media Thread.
- **Forensic Veracity Score (0-100)** rendered as a Plotly donut chart, colour-coded emerald / amber / red.
- **Multimodal Consistency** status badges for image, video, and audio inputs (Audio/Visual alignment · Lighting consistency · Artifact cleanliness).
- **Cross-Reference Engine** that fetches a URL's *main* content, strips nav / sidebar / comment noise, and matches canonical claims using **word-boundary matching** with a **relevance score** (0-1).
- **Hallucination & Forensic Warnings** that flag generative-AI tell-tales (e.g. *"a testament to"*, *"furthermore"*), long sentence patterns, low lexical diversity, and assertive absolute language.
- **Optional LLM Augmentation** — when `OPENAI_API_KEY` is set as a system environment variable, an extra panel adds per-claim neutral summaries and creative-leap detection. Silently falls back to heuristic-only mode otherwise.
- **Dark, ultra-modern UI** with a glassy card layout, gradient hero, pill-style radio toggle, and status badges throughout.

---

## 🚀 Quick start

### 1. Clone & enter the project
```bash
git clone <your-repo-url> omniguard-ai
cd omniguard-ai
```

### 2. Create the virtual environment and install dependencies

**Windows (PowerShell):**
```powershell
.\setup_env.bat
```

**Linux / macOS (Bash):**
```bash
bash setup_env.sh
```

Both scripts will:
- Create a `.venv` virtual environment (no global pollution)
- Upgrade `pip`
- Install everything from [`requirements.txt`](requirements.txt)

### 3. Run the dashboard

```bash
# Windows
.\.venv\Scripts\streamlit.exe run app.py

# Linux / macOS
.venv/bin/streamlit run app.py
```

Then open <http://localhost:8501> in your browser. **No API key is required** — the heuristic pipeline works on its own.

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

OmniGuard runs **two complementary pipelines**:

### 1. Heuristic forensic pipeline — [`utils/verifier.py`](utils/verifier.py)

| Content type | What it analyses                                                                                                                                                                                                                                                                            |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Text**     | Sentence / word counts, type-token ratio, punctuation & number density, LLM tell-tale phrases, assertive-language detection                                                                                                                                                                |
| **Image**    | Magic-byte detection (PNG / JPEG / GIF / WEBP / BMP), per-block Shannon entropy, block-variance and alignment-anomaly signals, simulated lighting / artifact scores                                                                                                                          |
| **Video**    | Container-header parsing (MP4 / MKV / AVI / MOV), byte-level entropy proxy, simulated frame / lighting / artifact scores                                                                                                                                                                    |
| **Audio**    | RIFF/WAVE header parsing for sample rate & channels, byte-level entropy, voice-consistency and artifact proxies                                                                                                                                                                              |
| **URL**      | Fetches the first 1 MB, extracts `<article>` → `<main>` → highest `<p>`-density block via BeautifulSoup, matches against a built-in fact table with word-boundary regex, and ranks hits by **relevance** (position × body length × occurrence count)                                       |

> **Why no PIL / ffmpeg / opencv?** Image / video / audio signals are derived from header + byte-level entropy only. This keeps the working set tiny and lets the app run comfortably on 8 GB RAM machines.

### 2. Optional LLM augmentation — [`utils/llm.py`](utils/llm.py)

- Reads the key with `os.getenv("OPENAI_API_KEY")` at every call — never from any project file.
- Adds per-claim neutral summaries and creative-leap detection.
- Every call is wrapped in silent-failure guards so a quota or network error **never** blocks the dashboard.
- A 60-second cached health probe distinguishes the three states shown above.

---

## 🏗️ Project structure

```
OmniGuard AI/
├── app.py                       # Streamlit dashboard (UI + orchestration)
├── components/                  # Reserved for reusable UI components
│   └── __init__.py
├── utils/
│   ├── __init__.py
│   ├── verifier.py              # Heuristic analysis pipeline
│   └── llm.py                   # Optional OpenAI augmentation
├── test_app_e2e.py              # End-to-end AppTest suite
├── test_relevance.py            # Cross-reference regression tests
├── requirements.txt
├── setup_env.bat                # Windows venv + dependency installer
├── setup_env.sh                 # Linux / macOS venv + dependency installer
├── .env.example                 # Template (no real keys, ever)
├── .gitignore                   # Excludes .venv, .env, secrets
└── README.md                    # ← you are here
```

---

## 🧪 Testing

The project ships with two test suites:

```bash
# End-to-end UI test - drives the real app via AppTest
# and exercises all three content-type paths
.\.venv\Scripts\python.exe test_app_e2e.py

# Regression test for the cross-reference engine
# (word-boundary matching, relevance scoring, main-content extraction)
.\.venv\Scripts\python.exe test_relevance.py
```

Both should print `All scenarios passed.` / `All regression tests passed.` when the system is healthy.

---

## 🛠️ Tech stack

- **Python 3.10+** (developed and verified on 3.14)
- **[Streamlit](https://streamlit.io/)** — UI framework
- **[Plotly](https://plotly.com/python/)** — donut chart, future visualisations
- **[Requests](https://requests.readthedocs.io/)** + **[BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)** — URL fetching & parsing
- **[OpenAI SDK](https://github.com/openai/openai-python)** (optional) — LLM augmentation
- **[python-dotenv](https://pypi.org/project/python-dotenv/)** — `.env` loading (no secrets stored in repo)

---

## 📋 Design constraints

- **8 GB RAM target** — no PIL, no ffmpeg, no opencv, no torch.
- **No secrets in the repo** — `.env` is in `.gitignore`; `.env.example` is the only template file. The OpenAI SDK is imported lazily.
- **Heuristic-first** — every LLM call has a silent-failure guard. The dashboard always renders, with or without a working API.
- **Explainable heuristics** — every score is built from inspectable measurements (TTR, entropy, magic bytes, container format, etc.), not black-box predictions.

---

## 🧩 Extending the pipeline

Adding a new verification module is straightforward:

1. Drop a new function in [`utils/verifier.py`](utils/verifier.py) that returns a dict shaped like the existing per-modality dicts (`multimodal` / `cross_reference_results` / `hallucination_report`).
2. Add a new entry in `CONTENT_TYPE_OPTIONS` in [`app.py`](app.py) and a corresponding `_render_*_tab` (or a new branch in `main()`).
3. Add a regression test in [`test_relevance.py`](test_relevance.py) or a new test file.

The 8 GB RAM rule still applies: prefer header / byte-level signals over pixel / frame decoding.

---

## 🤝 Contributing

PRs welcome. Please:
- Run both test suites before opening a PR
- Avoid adding heavy dependencies (PIL, opencv, torch) — there are usually header / byte-level shortcuts
- Never commit `.env` or any real API key

---

## 📄 License

MIT — see [`LICENSE`](LICENSE) (add this file before publishing if you want MIT).

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) for the dashboard framework
- [OpenAI](https://openai.com/) for the optional LLM enrichment path
- The wide community of LLM-detection researchers whose work informed the heuristic tell-tale list

---

<sub>Built with ❤️ for people who want a second opinion on what they read online.</sub>
