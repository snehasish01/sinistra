# HLJT dataset (Hand Laterality Judgement Task)

Trial-level data used by `py/examples/hljt_analysis.py`.

## Attribution

Moreno-Verdú M, McAteer SM, Waltzing BM, Van Caenegem E, Hardwick RM (2025).
Development and validation of an open-source Hand Laterality Judgement Task for
in-person and online studies. Neuroscience.
https://doi.org/10.1016/j.neuroscience.2025.02.056
Data from OSF project https://osf.io/8h7ec/, licensed CC BY 4.0.

## Files

Downloaded verbatim from OSF (stable file GUIDs):

| file | OSF download | rows |
| --- | --- | --- |
| `all_data_inperson.csv` | https://osf.io/download/6710d1133459409ef2210e78/ | 15,153 trials, 40 participants |
| `all_data_online.csv`   | https://osf.io/download/6710d1137d213047b6014863/ | 22,036 trials, 60 participants |

These are the analysis-ready trial-level tables (RT already trimmed to
[300, 3000] ms, practice block removed, `Reject_trial == "no"` throughout).
Only `all_data_inperson.csv` is used in the Phase 7 analysis.

Key columns: `ID`, `Group` (response mode), `Accuracy` (0/1), `Side`
(depicted hand / correct answer), `Direction` (`Up`/`Medial`/`Lateral`/`Down`
rotation sense), `View` (`Palmar`/`Dorsal`), `Angle` (0–315°, hand-anatomical
frame), `Angle_unif` (disparity from upright: 0/45/90/135/180), `RT` (ms;
`Response_time` is a byte-identical duplicate), `Block`. The unnamed first
column is an `R` rowname artifact.
