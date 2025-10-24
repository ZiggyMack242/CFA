# CFA
a working meta-epistemic lab: you can tune assumptions and watch transparency move.

# CFA v2.0 Interactive Console

Comparative Framework Audit tool with interactive toggles and visual comparisons.

Core Features (from Nova’s spec):
✅ All 4 toggles (Parity, PF-Type, Fallibilism, BFI-Weight)
✅ Dual framework editor (MdN and CT preloaded)
✅ YPA Trinity calculation (Neutral/Existential/Empirical)
✅ Guardrails implementation (Lever-Coupling, BFI-Sensitivity)
✅ Symmetry Audit with Δ flagging
Enhanced Features I Added:
✨ Interactive Plotly charts (lever comparison + YPA trinity overlay)
✨ Tabbed interface for clean organization (Visual/Detailed/Guardrails/Symmetry)
✨ “Admits Limits” checkbox per framework (controls fallibilism bonus eligibility)
✨ Winner summary table with deltas across scenarios
✨ JSON export of full configuration + results
✨ Help tooltips on all toggles
✨ Color-coded pass/fail indicators (✅⚠️)

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
