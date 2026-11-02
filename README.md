# FairSpeech

**Measuring and Mitigating ASR (Automatic Speech Recognition) Bias Propagation in AI-Based Interview Feedback for Non-Native Speakers**

FairSpeech measures how often speech-recognition (ASR) errors turn into unfair feedback for speakers with different accents (RQ1), and trains a feedback gate that suppresses ASR-induced false feedback, compared with a confidence-threshold baseline (RQ2). The project uses the Edinburgh International Accents of English Corpus (EdAcc) and includes a Streamlit interview-practice demo.

Proposal and plan: see [`docs/proposal/`](docs/proposal/).

## Team

| Member | Name | Role |
|---|---|---|
| A | Jaswanth Maradani | Data & Statistics lead |
| B | Sanjay Eshwaradurai Vasantthisri | ASR lead |
| C | Dize Demirtas | Feedback & App lead |
| D | Okojie Juliet | Project Management & Documentation lead |

## Repository structure

```
fairspeech/
├── data/                # not committed, except data/mock/ (see data/README.md)
│   ├── raw/             # EdAcc download
│   ├── interim/         # segmented audio and segment table
│   ├── processed/       # ASR outputs, feedback JSON, labels
│   └── mock/            # small example files for tests and the app
├── docs/                # shared formats, minutes, MO2 (D), proposal and plan
├── schemas/             # JSON schema for the feedback format
├── notebooks/           # exploration (Colab)
├── outputs/             # figures and result tables (not committed)
├── src/
│   ├── data/            # A: EdAcc loading, segmentation
│   ├── asr/             # B: Whisper, CrisperWhisper, Vosk, WER
│   ├── gate/            # B: gate features, threshold baseline, LR/LightGBM training, evaluation
│   ├── feedback/        # C: grammar, fluency, filler, alignment, metrics
│   ├── analysis/        # A: RQ1/RQ2 statistics, figures
│   └── app/             # C: Streamlit app
└── tests/
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pytest                                                # checks the mock feedback JSON
streamlit run src/app/streamlit_app.py
```

LanguageTool needs Java 17 or later. ASR models run on Colab: `pip install -r requirements-asr.txt`.

## Portfolio contents

*Annotated table of contents with contributors, filled in by D during the project.*

| Item | Description | Contributor(s) |
|---|---|---|
| [Contributing guide](CONTRIBUTING.md) | Branch rules, review rotation, commit conventions | C |
| [Data contracts](docs/data_contracts.md) | Segment table and feedback JSON formats | A (segment table), C (feedback JSON) |
| [Proposal and plan](docs/proposal/) | Project proposal, final plan and team roles | team |
| [ASR inference & evaluation](src/asr/README.md) | Whisper, CrisperWhisper, Vosk wrappers, WER, normalisation | B |
| [Feedback gate](src/gate/README.md) | Feature extraction, baseline sweep, LR/LightGBM training, evaluation | B |
| [ASR & Gate Colab Notebook](notebooks/asr_and_gate_colab.ipynb) | GPU transcription & feedback gate training workflow | B |
