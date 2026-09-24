# calibrated-refusal-eval

A personal open-source evaluation harness for **answer | refuse | escalate** decision calibration in medical RAG systems.

## Overview

When a medical information system retrieves context snippets to answer patient questions, it must decide whether to:

- **answer**: The evidence is sufficient and safe to provide information
- **refuse**: Insufficient evidence, out-of-scope, or inappropriate request
- **escalate**: Urgent/emergent situation requiring immediate professional consultation

This harness evaluates three approaches across **~96 synthetic scenarios**:

- **OpenAI arm**: Chat model with prompted label + elicited confidence (0–1)
- **Gemini arm**: Chat model with same prompt protocol as OpenAI
- **Jev arm**: Decision model with native choice probabilities via Vercel AI Gateway

## Thesis

**Refusal needs calibration, not vibes.** A system that confidently refuses when it should escalate—or answers when it should refuse—creates safety and trust issues. This harness measures calibration via accuracy, ECE, Spearman correlation, and pairwise Wilcoxon tests.

**Key Question:** Do chat models with elicited confidence differ from decision models with native probabilities on calibration metrics? This harness lets you test both approaches with statistical rigor.

---

## What This Project Is

- **Personal OSS toolkit** for evaluating medical RAG decision-making
- **~96 synthetic scenarios** (no real patient data / PHI)
- **Unified decision policy** used by all arms (see `POLICY.md`)
- **Statistical rigor** via `reliability-eval` (Wilcoxon signed-rank, Spearman rank correlation)
- **3-way comparison**: OpenAI vs Gemini vs Jev

## What This Project Is NOT

- ❌ Not affiliated with any employer or healthcare organization
- ❌ Not a clinical tool or medical device
- ❌ Not trained on or evaluated with real patient data
- ❌ Not a peer-reviewed academic paper
- ❌ Not making strong claims about model superiority (small n=96, synthetic data, known limitations)

**Use Responsibly:** This is an educational/research project for personal learning and public demonstration of calibration measurement techniques. Results on synthetic data do not generalize to production systems without further validation.

---

## Dataset

**Current version:** ~96 items  
**Distribution:** ~48% answer, ~11% refuse, ~41% escalate

All items are synthetic medical Q+A scenarios designed to stress-test answer/refuse/escalate boundaries. Covers domains: medication, diagnostics, symptoms, procedures, nutrition, vaccination, chronic disease, mental health, pediatrics, geriatrics, and more.

See `DATA.md` for full dataset documentation and adjudication log.

---

## Installation

```bash
npm install
```

---

## Quick Start (Dry Run)

Run the evaluation with deterministic predictions (no API keys required):

```bash
npm run eval:dry
```

This uses ~96 synthetic scenarios and heuristic-based stubs for all three arms.

---

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# OpenAI configuration (Arm A: OpenAI with elicited confidence)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-6-sol

# Gemini configuration (Arm B: Gemini with elicited confidence)
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.1-flash-lite

# LLM temperature (applies to both OpenAI and Gemini)
LLM_TEMPERATURE=0.0

# Vercel AI Gateway (Arm C: Jev with native probabilities)
AI_GATEWAY_API_KEY=...

# Legacy Jev endpoint (optional fallback)
JEV_API_KEY=
JEV_ENDPOINT=
```

**3-Way Setup:**
- **OpenAI**: Set `OPENAI_API_KEY` (uses configured model, e.g., gpt-6-sol)
- **Gemini**: Set `GEMINI_API_KEY` (uses configured model, e.g., gemini-3.1-flash-lite)
- **Jev**: Set `AI_GATEWAY_API_KEY` (Vercel AI Gateway - no waitlist required)

**Live Mode Requirements:**
- At least 2 API keys must be configured
- Ideal: all 3 keys for full 3-way comparison

**Dry-Run Mode:**
- No API keys required
- Always runs all three arms with deterministic stubs

---

## Running Live Evaluation

### Full 3-Way Comparison (Recommended)

1. Configure all three API keys in `.env`
2. Run live evaluation:

```bash
npm run eval
```

This will run:
- **OpenAI arm**: Configured model (e.g., gpt-6-sol) with elicited confidence
- **Gemini arm**: Configured model (e.g., gemini-3.1-flash-lite) with elicited confidence (same prompt as OpenAI)
- **Jev arm**: Native probability distributions via Vercel AI Gateway

The harness computes pairwise Wilcoxon tests for all three pairs:
- OpenAI vs Gemini
- OpenAI vs Jev
- Gemini vs Jev

### 2-Way Comparison

Set any two of the three API keys. Live mode requires at least 2 arms. If fewer than 2 keys are present, the harness will fail with a clear error message listing missing keys.

---

## Metrics

The harness computes:

- **Accuracy**: Correct label prediction rate (per arm)
- **ECE** (Expected Calibration Error): Deviation between confidence and actual accuracy (per arm)
- **Brier Score**: Mean squared error of probability predictions (per arm, **see note below**)
- **Per-label accuracy**: Performance breakdown by answer/refuse/escalate (per arm)
- **Spearman ρ**: Confidence-correctness correlation (per arm, powered by `reliability-eval`)
- **Pairwise Wilcoxon signed-rank tests**: Statistical comparison between every pair of arms (powered by `reliability-eval`)

### ⚠️ Important Metric Note: Brier Score Fairness

**Multiclass Brier Score is NOT directly comparable between chat arms (OpenAI, Gemini) and Jev:**

- **Chat arms** elicit a single confidence value and fabricate 3-way probabilities as `(conf, (1-conf)/2, (1-conf)/2)`. This is not a true multiclass distribution.
- **Jev arm** outputs native probability distributions over all three choices.

**Recommendation:** For fair comparison across arms, prefer **Accuracy**, **ECE**, and **Spearman ρ** (confidence-correctness correlation). Brier score is reported but should be interpreted with caution for chat arms.

### 3-Way Pairwise Comparisons

When all three arms run, the harness computes Wilcoxon tests for all three pairs:

1. **OpenAI vs Gemini**: Do the two chat models differ significantly?
2. **OpenAI vs Jev**: Does elicited confidence (OpenAI) differ from native probabilities (Jev)?
3. **Gemini vs Jev**: Does elicited confidence (Gemini) differ from native probabilities (Jev)?

Each comparison reports:
- **p-value**: Statistical significance (lower = more significant)
- **Significant (α=0.05)**: Yes/No at 5% significance level
- **Effect size**: Magnitude of difference (standardized)

### Statistical Tests via reliability-eval

This project uses [`reliability-eval`](https://github.com/ahmadRahman1993/reliability-eval) for rigorous statistical comparison:

- **Wilcoxon signed-rank test**: Paired non-parametric test comparing accuracy on the same evaluation items. Reports p-value, significance (α=0.05), and effect size.
- **Spearman rank correlation**: Measures whether higher confidence correlates with higher correctness. Positive ρ indicates confidence is a useful signal for routing decisions.

Both tests are validated against scipy.stats reference implementations.

---

## Project Structure

```
├── src/
│   ├── types.ts              # TypeScript type definitions
│   ├── policy.ts             # Unified decision policy (shared by all arms)
│   ├── data/
│   │   └── seed.json         # ~96 synthetic evaluation items
│   ├── arms/
│   │   ├── llm.ts            # OpenAI & Gemini arms (elicited confidence)
│   │   └── jev.ts            # Jev arm (native probabilities)
│   └── eval/
│       ├── metrics.ts        # Calibration metrics computation
│       └── run.ts            # Main 3-way evaluation runner
├── POLICY.md                 # Unified decision policy documentation
├── DATA.md                   # Dataset documentation and adjudication log
├── package.json
├── tsconfig.json
├── .env.example
└── README.md
```

---

## Decision Policy

All evaluation arms use a **unified decision policy** defined in `POLICY.md`. Key principles:

1. **ANSWER**: Sufficient evidence + educational/factual + no patient-specific decision + no safety risk
2. **REFUSE**: Insufficient evidence OR out-of-scope OR inappropriate request
3. **ESCALATE**: Urgent/emergent OR patient-specific medication decision OR complex case OR ongoing symptoms requiring workup

**When in doubt on safety issues, prefer ESCALATE.**

The same policy text is used in:
- Chat system prompts (OpenAI, Gemini)
- Jev instructions and criteria fields

This ensures fair comparison across arms.

---

## Privacy & Scope

- **Synthetic data only**: No real patient information (PHI)
- **Educational purpose**: Medical scenarios are illustrative examples
- **Personal OSS project**: Not affiliated with any employer or healthcare organization

---

## Known Limitations

1. **Small Sample Size**: n=96 is sufficient for initial calibration measurement but not for robust subgroup analysis or fairness evaluation across patient demographics.

2. **Synthetic Scenarios**: All items are synthetic and designed to stress-test decision boundaries. Real-world question distributions will differ. Calibration measured here may not transfer to production.

3. **Dry-Run Heuristics**: Dry-run stubs use deterministic heuristics and are not designed to be well-calibrated. Real results require live API calls.

4. **No Grounding Evaluation**: We assume retrieved snippets are accurate and relevant. In production, poor retrieval degrades decision quality.

5. **Single-Turn Evaluation**: Items are single-turn Q+A. Real medical AI often uses multi-turn clarification.

6. **Brier Score Comparability**: Chat arms fabricate multiclass probabilities from elicited confidence. Brier score is reported but not directly comparable to Jev.

---

## Results

Run `npm run eval` with live API keys to generate results. See `results.json` for full output.

**Interpreting Results:**
- **Accuracy**: Which arm makes correct decisions most often?
- **ECE**: Which arm is best calibrated (confidence matches accuracy)?
- **Spearman ρ**: Is confidence a useful signal for routing decisions (positive and significant)?
- **Wilcoxon p-values**: Are differences between arms statistically significant?

**⚠️ Caution:** Small n=96 limits statistical power. Non-significant results don't prove equivalence. Significant results may not generalize beyond synthetic data.

---

## License

MIT - See LICENSE file

---

## Author

Ahmad Rahman  

---

## Contributing

Issues and PRs welcome. Focus areas:

- Additional synthetic evaluation scenarios (balanced across answer/refuse/escalate)
- Live Jev-style API integrations (beyond Vercel AI Gateway)
- Calibration visualization tools (reliability diagrams, bin plots)
- Multi-turn conversation scenarios
- Retrieval quality evaluation
- Fairness evaluation across patient demographics (synthetic personas)

---

## Citation

If you use this harness or dataset, please cite:

```
Ahmad Rahman (2026). Calibrated Refusal Evaluation Harness.
Personal OSS project. https://github.com/ahmadRahman1993/calibrated-refusal-eval
```
