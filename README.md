# ✈️ Airline Customer Satisfaction — Predictive Analysis

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-00897B?style=flat-square)

> **Can we predict whether an airline passenger will be satisfied — and identify exactly what drives that satisfaction?**  
> This end-to-end data science project answers that question using 129,880 passenger records, statistical feature selection, and interpretable machine learning models.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Project Workflow](#-project-workflow)
- [Key Findings & Insights](#-key-findings--insights)
- [Machine Learning Results](#-machine-learning-results)
- [Business Recommendations](#-business-recommendations)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Author](#-author)

---

## 🎯 Project Overview

Airlines operate in a hyper-competitive market where **passenger experience directly defines loyalty and revenue**. This project builds a classification model to predict customer satisfaction and uncovers the service attributes that matter most.

**Business Goal:** Help airlines prioritize service investments by identifying the strongest drivers of passenger satisfaction.

**Analytical Goal:** Train and evaluate machine learning models that accurately classify passengers as *Satisfied* or *Dissatisfied*.

---

## 📊 Dataset Description

| Property | Value |
|---|---|
| Records | 129,880 passengers |
| Features | 22 (21 predictors + 1 target) |
| Target Variable | `Satisfaction` (Satisfied / Dissatisfied) |
| Class Balance | 54.7% Satisfied, 45.3% Dissatisfied |
| Missing Values | 393 (Arrival Delay column only) |
| Duplicates | None |

### Feature Categories

**Demographics & Trip Info**
- `Age` — Passenger age (7 to 85 years)
- `Customer Type` — Loyal Customer / Disloyal Customer
- `Type of Travel` — Business Travel / Personal Travel
- `Class` — Business / Eco / Eco Plus
- `Flight Distance` — Distance in kilometres

**14 Service Rating Features** *(rated 1–5)*
- Seat Comfort, Food & Drink, Inflight WiFi, Inflight Entertainment
- Online Boarding, Online Support, Ease of Online Booking
- On-board Service, Leg Room Service, Baggage Handling
- Checkin Service, Cleanliness, Gate Location
- Departure/Arrival Time Convenience

**Operational Features**
- `Departure Delay in Minutes`
- `Arrival Delay in Minutes`

---

## 🔄 Project Workflow

```
Data Loading → Cleaning → EDA → Outlier Treatment → Feature Selection → Preprocessing → ML Modelling → Evaluation
```

### 1. 🧹 Data Cleaning
- Removed duplicate rows (none found in this dataset)
- Converted target variable to binary: `satisfied → 1`, `dissatisfied → 0`
- Imputed 393 missing values in `Arrival Delay in Minutes` using **median** (robust to skew)

### 2. 📈 Exploratory Data Analysis
- Bar charts for all 17 categorical predictors
- Histograms for 4 continuous predictors: Age, Flight Distance, Departure Delay, Arrival Delay
- Grouped bar plots to visualise categorical predictor vs. target relationships
- Box plots for continuous predictor vs. target relationships

### 3. 🔧 Outlier Treatment (Capping Method)
| Feature | Threshold | Action |
|---|---|---|
| Flight Distance | > 5,000 km | Capped at 4,996 |
| Departure Delay | > 200 min | Capped at 199 |
| Arrival Delay | > 200 min | Capped at 199 |

### 4. 📐 Statistical Feature Selection
All 21 features were statistically validated before modelling:

- **ANOVA Test** (Categorical Target vs. Continuous Predictors) — All 4 continuous features selected (p < 0.05)
- **Chi-Square Test** (Categorical Target vs. Categorical Predictors) — All 17 categorical features selected (p < 0.05)

> ✅ **All 21 features passed** — no feature was rejected at this stage.

### 5. ⚙️ Data Preprocessing
- Binary nominal encoding: `Type of Travel` mapped to 0/1
- One-hot encoding via `pd.get_dummies()` for multi-class nominal variables (`Customer Type`, `Class`)
- Feature scaling: **Min-Max Normalization** (outperformed StandardScaler on this data)
- Train/Test split: **70% training / 30% testing** (stratified, `random_state=42`)

---

## 🔍 Key Findings & Insights

### Passenger Segmentation

| Segment | Satisfaction Rate |
|---|---|
| Loyal Customer — Business Travel | ~78% |
| Loyal Customer — Personal Travel | ~41% |
| Disloyal Customer — Business Travel | ~28% |
| Disloyal Customer — Personal Travel | ~8% |

> 💡 **Business travellers are 3× more satisfied than personal travellers.**  
> Loyal customers are 5.6× more likely to be satisfied than disloyal ones.

### Top Satisfaction Drivers

Based on Decision Tree feature importance and Chi-Square test results:

1. 🥇 **Online Boarding** — #1 Decision Tree split point
2. 🥈 **Inflight Entertainment**
3. 🥉 **Seat Comfort**
4. **Inflight WiFi Service**
5. **Type of Travel**
6. On-board Service, Leg Room, Food & Drink, Ease of Online Booking, Class

### Service Strengths vs. Weaknesses

| ✅ Strengths | ⚠️ Gaps to Address |
|---|---|
| Baggage Handling | Inflight WiFi (top complaint) |
| On-board Service | Ease of Online Booking |
| Checkin Service | Gate Location convenience |
| Inflight Entertainment | Departure/Arrival Timing |

---

## 🤖 Machine Learning Results

### Model Comparison

| Metric | Logistic Regression | Decision Tree ⭐ |
|---|---|---|
| Test Accuracy | 83% | **85%** |
| 10-Fold CV Score | 77% | **80%** |
| Dissatisfied — Precision | 81% | **90%** |
| Dissatisfied — Recall | 81% | 74% |
| Satisfied — Precision | 84% | 81% |
| Satisfied — Recall | 84% | **93%** |

**Winner: Decision Tree** (`max_depth=3`, `criterion='entropy'`)  
Reasons: Higher accuracy, better cross-validation score, and produces interpretable IF-THEN rules for business stakeholders.

### Decision Tree Logic (Top Rules)

```
IF Online Boarding ≤ 3:
    IF Type of Travel = Personal → DISSATISFIED
    IF Type of Travel = Business → DISSATISFIED (lower threshold)
ELSE (Online Boarding > 3):
    IF Seat Comfort ≥ 4 → SATISFIED
    IF Seat Comfort < 4 → DISSATISFIED
```

> The model is explainable — ideal for presenting to non-technical airline management.

---

## 💡 Business Recommendations

**01 — Fix the Online Boarding Experience**  
The #1 decision tree split. Invest in app UX, boarding gate notifications, and streamlined check-in. Even a small improvement here has a disproportionate impact on overall satisfaction scores.

**02 — Upgrade WiFi & Inflight Entertainment**  
Both rank in the top 5 satisfaction drivers. Passengers now treat reliable in-flight connectivity as a baseline expectation, not a premium perk.

**03 — Prioritize Loyal Customer Retention**  
Loyal customers represent 81.7% of volume. A 5% improvement in their satisfaction rate could significantly transform repeat bookings and revenue.

**04 — Engage Personal Travellers with Value**  
Only 24% of personal travellers are satisfied — a massive untapped segment. Targeted perks like seat upgrades, lounge access bundles, or loyalty onboarding could convert this group.

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core programming language |
| Pandas | Data manipulation & analysis |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualisation |
| Scikit-learn | Machine learning models, preprocessing, evaluation |
| SciPy | ANOVA and Chi-Square statistical tests |
| Jupyter Notebook | Interactive development environment |

---


---

## 📁 Project Structure

```
airline-customer-satisfaction/
├── Airline_Customer_Satisfaction.ipynb          # Main analysis notebook
├── Airline_customer_satisfaction (PPT).pdf      # Raw dataset (add here)
└── README.md                                    # This file
```

---

## 👤 Author

**[Sweta Mehta]**  

Data Science Portfolio Project  


