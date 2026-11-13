# Anticipating PM2.5 Episode Onsets in Beijing: A LightGBM Early-Warning Proof of Concept with Leave-One-Station-Out Validation

**National Economics University | Faculty of Business Analytics**

**Data Mining – Group 5 | November 2026**

![Python](https://img.shields.io/badge/Python-3.x-blue) ![LightGBM](https://img.shields.io/badge/LightGBM-gradient%20boosting-green) ![Status](https://img.shields.io/badge/Status-academic%20proof%20of%20concept-orange)

> **Disclaimer:** This project is an academic proof of concept for PM2.5 episode-onset early warning. It is **not** an operational warning system and must not be used for public-health or regulatory decisions. Results come from one city and one test year.

## 🔗 Quick Access

- 📄 [Read the Final Report](Group5_Final_Report.pdf)
- 💻 [Explore the Research Notebook](DataMining_Group5_AirPollutantEarlyWarning.ipynb)
- 📊 [Browse Figures and Tables](outputs/)

---

## 1. Project Overview

Average PM2.5 concentrations in Beijing fell sharply between 2013 and 2017, yet severe pollution episodes persist. Most PM2.5 forecasting studies predict **concentrations**; far fewer ask whether the **onset of a sustained episode** can be anticipated while PM2.5 is still below the threshold.

This project builds and evaluates an episode-onset early-warning framework from hourly station observations, and tests whether the model transfers to monitoring stations it has never seen.

**Dataset:** Beijing Multi-Site Air-Quality Data (UCI Machine Learning Repository)

| Property | Description |
|---|---|
| Stations | 12 Beijing monitoring stations |
| Period | 1 March 2013 – 28 February 2017 |
| Hourly records | 420,768 station-hours (35,064 per station) |
| Target | PM2.5 (µg/m³) |
| Co-pollutants | PM10, SO2, NO2, CO, O3 |
| Weather | Temperature, pressure, dew point, rainfall, wind speed, wind direction |
| Missing values | Kept as missing, no imputation |

**Episode definition:** a run of at least **6 consecutive hours** with PM2.5 **> 75 µg/m³** (the 24-hour Grade II limit in China's ambient air quality standard, applied here to hourly values). The **onset** is the first hour of the run.

**Prediction task:** at each station and hour, while PM2.5 ≤ 75 µg/m³ and no episode is in progress, predict whether an episode will begin within the next **H hours** (H = 1, 3, 6, 12, 24).

### Key Findings

- **Current PM2.5 carries most of the signal.** The pre-specified hypothesis that co-pollutants and weather *without* PM2.5 (scenario B) would beat a PM2.5-only model was **not supported** (episode recall at H = 6 h: 0.70 vs 0.90).
- **Weather adds modest, horizon-dependent value.** The selected PM2.5 + weather model (PW) at H = 12 h reached a usable recall of **0.81** with **11.3 false alarms per station per 30 days**, only **+0.047** above PM2.5-only in an exploratory comparison.
- **The model transferred to unseen stations with a small loss.** In leave-one-station-out (LOSO) validation, usable recall fell by only 0.013 and all 12 stations passed the transfer gate.
- **Limits:** one city, one test year, and results depend on the 75 µg/m³ threshold and the false-alarm budget.

---

## 2. Research Questions

**Master Research Question:** Can historical data on time, meteorological conditions and air pollutants identify the onset of high-PM2.5 episodes, and can the model generalize across monitoring stations?

| RQ | Focus | Question |
|---|---|---|
| RQ1 | Observe | When and at which stations do PM2.5 levels peak, and when do episodes occur? |
| RQ2 | Explain | Which co-pollutants and weather variables are most associated with PM2.5? |
| RQ3 | Discover | Do distinct weather regimes exist, and do they separate episode onsets? |
| RQ4 | Anticipate | Can episode onset be anticipated within the next H hours, and how does performance change with H? |
| RQ5 | Generalize | Does the model work at stations not in the training data? |

**Pre-specified hypothesis (RQ4):** a model using co-pollutants and weather without PM2.5 outperforms a PM2.5-only model in both PR-AUC lift and recall under the same false-alarm budget.

---

## 3. Methodology

### 3.1 Data Preparation
- Complete hourly grid per station; no rows removed, no values imputed.
- Seasons follow calendar months (winter Dec–Feb, spring Mar–May, summer Jun–Aug, autumn Sep–Nov).
- Two gap-handling rules: descriptive analyses (RQ1–RQ3) merge runs separated only by ≤ 3 h of missing data; prediction tasks (RQ4–RQ5) are labelled per station, do not bridge missing data, and exclude "uncertain" episodes. **Episode counts are therefore not comparable between the two parts.**

### 3.2 Analysis by Research Question

| RQ | Methods |
|---|---|
| RQ1 | Mean / median / P90 by month and hour; regional episode profile; station differences with 14-day block bootstrap |
| RQ2 | Spearman correlation (pooled, per station, lagged 0–24 h); rank-biserial effect size and Mann–Whitney U for high vs other hours |
| RQ3 | K-means on 5 standardized weather variables (K = 2–8, silhouette with ≥ 5% minimum cluster size); stability via adjusted Rand index; Cramér's V with season; onset ratio with 5-day block bootstrap |
| RQ4 | LightGBM classifiers, 7 input scenarios, 5 horizons, chronological split, episode-level metrics |
| RQ5 | Leave-one-station-out validation compared with the pooled model |

### 3.3 Modelling Scenarios (RQ4)

| Scenario | Inputs at time t (past-only) |
|---|---|
| Climatology | Onset frequency by station × month × hour (no model) |
| PM2.5-only | PM2.5 at t and t−1 |
| A | Co-pollutants |
| W | Weather and calendar variables |
| B | A + W |
| PW | W + PM2.5 |
| C | B + PM2.5 lags and changes |

All models use the same LightGBM algorithm with early stopping on validation PR-AUC. **No station identifier is used**, so models can be applied to unseen stations.

### 3.4 Experimental Configuration

| Parameter | Configuration |
|---|---|
| Training period | 1 Mar 2013 – 28 Feb 2015 |
| Validation period | 8 Mar 2015 – 29 Feb 2016 |
| Test period | 8 Mar 2016 – 28 Feb 2017 (single final evaluation) |
| Gaps between periods | 7 days (longer than the largest horizon) |
| Horizons | H = 1, 3, 6, 12, 24 h (primary H = 6 h) |
| False-alarm budget | ≤ 12 per station per 30 days (operator assumption) |
| Cost ratio (reporting only) | 10 : 1 (missed episode : false alarm) |
| Uncertainty | Paired 5-day block bootstrap, 2,000 resamples |
| Random seed | 42 |

### 3.5 Evaluation Metrics
- **PR-AUC and PR-AUC lift** over same-horizon climatology
- **Episode recall:** share of episodes preceded by at least one warning ≥ 1 h before onset
- **Usable recall:** share of episodes warned **≥ 3 h** ahead
- **False alarms** per station per 30 days
- **Median lead time**

### 3.6 Model Selection and Deployment Gates
Candidates (PM2.5-only, W, PW, B, C at H = 6 and 12 h) are ranked on **validation** usable recall. Among candidates within 0.02 of the best, the one needing the fewest data sources is selected. On the test year it must pass three gates:

1. **G1:** usable recall above climatology (lower CI bound of the difference > 0)
2. **G2:** false alarms ≤ 15 per station per 30 days (1.25 × budget)
3. **G3:** median lead time ≥ 3 h

### 3.7 Leakage Controls
Lag-direction check, future-data replacement test, and a gap longer than the largest horizon between training and validation. Scenario C versus B serves as an additional leakage check.

---

## 4. Main Experimental Results

### 4.1 EDA (RQ1–RQ3)
- 463 regional episodes; median duration 19 h (P90 66 h, longest 139 h).
- Monthly mean PM2.5 peaked in December (105 µg/m³) and was lowest in August (53 µg/m³).
- Episodes began in all 12 months; 49% started October–March (85% for episodes peaking ≥ 250 µg/m³).
- Station means ranged from 66.0 (Dingling) to 86.2 µg/m³ (Dongsi).
- PM2.5 correlated strongly with PM10 (ρ = 0.89, partly by construction), CO (0.84), NO2 (0.66) and SO2 (0.50); all weather variables had |ρ| ≤ 0.33.
- Two weather regimes (K = 2) were closely tied to season (Cramér's V = 0.71) but did **not** separate onsets (onset ratios 0.93 and 1.06).

### 4.2 Test Performance at H = 6 h (RQ4)

| Model | PR-AUC lift | Episode recall | False alarms / station / 30 d | Median lead (h) |
|---|---|---|---|---|
| Climatology | 1.00 | 0.29 | 11.0 | 4 |
| PM2.5-only | 2.39 | 0.90 | 11.4 | 2 |
| A (co-pollutants) | 1.87 | 0.49 | 9.9 | 3 |
| W (weather + calendar) | 1.92 | 0.57 | 12.4 | 3 |
| B (A + W) | 2.34 | 0.70 | 11.2 | 4 |
| PW (W + PM2.5) | 2.80 | 0.90 | 11.5 | 3 |
| C (B + PM2.5) | 2.82 | 0.87 | 10.9 | 3 |

Primary comparison B − PM2.5-only: recall −0.196 [−0.253, −0.146] → hypothesis **not supported**.

### 4.3 Effect of Horizon
As H grows from 1 h to 24 h, the positive rate rises from 0.020 to 0.369, PM2.5-only lift falls from 9.43 to 1.50, and the advantage of extra predictors narrows (W reaches the same lift as PM2.5-only at H = 24 h).

### 4.4 Selected Model: PW at H = 12 h (test year)

| Metric | PW | PM2.5-only | Climatology |
|---|---|---|---|
| Usable recall (≥ 3 h) | 0.81 [0.77, 0.84] | 0.76 [0.72, 0.80] | 0.39 [0.33, 0.45] |
| False alarms / station / 30 d | 11.3 | 11.2 | 10.8 |
| Median lead time (h) | 6 | 5 | 7 |
| PR-AUC | 0.561 | 0.505 | 0.284 |

PW passed all three gates. Its advantage over PM2.5-only (+0.047 usable recall) is an **exploratory** comparison and was clearest when a minimum lead of 2–8 h was required. Feature-group importance: PM2.5 (0.244 PR-AUC drop) ≫ calendar (0.041) > wind direction (0.030).

### 4.5 Transfer to Unseen Stations (RQ5)

| Comparison | Result |
|---|---|
| LOSO − pooled usable recall (PW, H = 12 h) | −0.013 [−0.025, −0.001] (0.80 vs 0.81) |
| LOSO − pooled false alarms | −0.20 [−0.39, −0.01] |
| Stations passing transfer gate | 12 / 12 |
| LOSO false alarms range | 9.8 – 14.2 (Huairou 14.2 exceeds the budget of 12 but stays below 15) |
| Adding `station_id` | No difference in recall, false alarms or PR-AUC |

LOSO tests transfer **between stations within Beijing**; it is likely more optimistic than transfer to another city.

---

## 5. Repository Structure

```
Beijing-PM25-Early-Warning/
│
├── README.md
│
├── DataMining_Group5_AirPollutantEarlyWarning.ipynb   # full workflow: EDA, RQ1–RQ5
│
├── Group5_Final_Report.pdf                          # final report
│
├── data/                                              # 12 UCI CSV files (see Section 6)
│   └── PRSA_Data_<Station>_20130301-20170228.csv
│
└── outputs/                                           # generated by the notebook
    ├── fig1_workflow … fig11_cases (.pdf, .png)
    ├── table1_data … table14_loso_H24 (.csv, .tex, .png)
    ├── key_numbers.txt
    ├── captions_rq45.md
    ├── deploy_model.txt                               # LightGBM model (PW, H = 12 h)
    └── deploy_config.json                             # model card: threshold, features, limitations
```

---

## 6. Reproducing the Research

**Requirements:** Python 3, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `lightgbm`.

```bash
git clone https://github.com/Thuy1310/Beijing-PM25-Early-Warning.git
cd Beijing-PM25-Early-Warning
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pandas scipy scikit-learn matplotlib lightgbm jupyter
```

**Data:** download the Beijing Multi-Site Air-Quality dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) and place the 12 files named `PRSA_Data_<Station>_20130301-20170228.csv` in the folder set by `DATA_DIR` in the notebook (default `"."`).

**Run:** open `DataMining_Group5_AirPollutantEarlyWarning.ipynb` in Jupyter or Google Colab and run all cells in order. Figures, tables and the model card are written to `Outputs`.

**Quick check:** set the environment variable `RQ45_QUICK=1` to run the RQ4–RQ5 section with far fewer bootstrap resamples. This is for testing the code only; reported numbers use 2,000 resamples. Full runs involve many model fits and bootstrap loops, so expect a long runtime.

---

## 7. Model Card and Deployment Artifacts

The notebook exports the selected model (PW, H = 12 h) as `outputs/deploy_model.txt` and a model card as `outputs/deploy_config.json`, containing the episode definition, alert threshold, required inputs, operating assumptions and limitations.

- **Alert rule:** warn when the score reaches the validation threshold; then stay silent for H hours at that station.
- **Run-time inputs:** hourly PM2.5 and weather (TEMP, PRES, WSPM, wind direction) over the last 48 h.
- **No warning** is issued when current PM2.5 is already above 75 µg/m³ or the last 48 h are not fully observed.
- A **streaming check** confirms that predictions from a rolling 48-hour window reproduce batch predictions.

These artifacts document how the model *could* be operated; they do not constitute a deployed service.

---

## 8. Limitations and Responsible Use

- **One city, one test year:** generalization to other years or regions is not confirmed.
- **Design choices matter:** the 75 µg/m³ threshold, the 6-hour minimum duration, the false-alarm budget (12 per station per 30 days) and the 10 : 1 cost ratio are assumptions, not empirical costs. Changing them could alter model rankings.
- **Station observations only:** no boundary-layer height, upstream transport data or weather forecasts. Missed onsets and transient false alarms may need such information (a hypothesis from three illustrative cases, not tested statistically).
- **Exploratory comparisons:** only two comparisons were pre-specified (B vs PM2.5-only; deployment selection rule); the rest are not corrected for multiple comparisons.
- **Associations, not causation.**
- **Label proximity:** episodes are defined from PM2.5 itself, so current PM2.5 near the threshold is naturally highly informative.
- **LOSO optimism:** all 12 stations lie in the same city and observe the same events.
- **Re-validation needed:** emission structure changes over time; the model should be re-validated on new data before any real use.

---

## 9. Team Members

National Economics University – Data Mining, Group 5

| Member | Student ID | Main contributions |
|---|---|---|
| Nguyen Nguyet Anh | 11230733 | Research questions; RQ4–RQ5 analysis plan and interpretation; EDA for RQ1–RQ3; Results section |
| Vu Khanh Linh | 11230760 | Literature review and research gaps; RQ2–RQ3; Introduction, Related Work, Discussion |
| Vu Phuong Anh | 11230736 | Methods section; RQ1, RQ4, RQ5 analyses; presentation slides |
| Trần Thu Thủy | 11230780 | RQ2–RQ3 co-development; Abstract and Conclusion; manuscript review; presentation slides |

**Instructor:** Nguyen Trong Nghia

---

## 10. Project Resources

| Resource | Link |
|---|---|
| Research notebook | [Open Notebook](DataMining_Group5_AirPollutantEarlyWarning.ipynb) |
| Final research report | [View PDF](Group5_Final_Report.pdf) |
| Dataset (UCI) | [Beijing Multi-Site Air-Quality Data](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) |
| Figures and tables | [outputs/](outputs/) |

**External documentation:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/) · [LightGBM](https://lightgbm.readthedocs.io/) · [scikit-learn](https://scikit-learn.org/)

For references, detailed methodology and discussion, see the final report.

---

*Data Mining – Group 5 | National Economics University | 2026*

*Developed for academic research and reproducible machine-learning experimentation.*
