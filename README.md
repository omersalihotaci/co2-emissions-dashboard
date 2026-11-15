# 🌍 Global CO₂ Explorer: Emissions, Energy & Economy

**CEN445 Introduction to Data Visualization: Assignment 2026–27**

An interactive Streamlit dashboard that explores **who emits CO₂, from which fuels, and how emissions relate to wealth and energy use** for 219 countries between 1990 and 2024. It also uses two machine-learning models to dig deeper.

**🔗 Live app:** https://co2-emissions-dashboardgit.streamlit.app/

---

## 📊 Dataset

| | |
|---|---|
| **Name** | CO₂ and Greenhouse Gas Emissions |
| **Publisher** | Our World in Data (OWID) |
| **Source** | https://github.com/owid/co2-data (file: `owid-co2-data.csv`) |
| **Underlying sources** | Global Carbon Project, Energy Institute Statistical Review, Maddison Project (GDP), UN population |
| **Raw size** | 56,269 rows × 79 columns (1750–2025, countries + regional aggregates) |
| **Used** | 7,867 rows × 28 columns after cleaning (219 countries, 1990–2025) |
| **Licence** | CC BY 4.0 |

Continent labels come from OWID's *Continents according to Our World in Data* table (`data/continents.csv`, https://ourworldindata.org/grapher/continents-according-to-our-world-in-data).

**Main variables used:** country, year, ISO code, continent (categorical) · population, GDP, total CO₂, CO₂ per capita, CO₂ per GDP, share of global CO₂, CO₂ from coal / oil / gas / cement / flaring, energy per capita, methane & N₂O per capita (numerical).

### Preprocessing (`preprocess.py`)
1. **Removed aggregate rows** (World, continents, income groups, "International shipping", and so on). They have no ISO code (7,963 rows).
2. **Restricted to 1990+**, the period with reliable coverage.
3. **Joined continents** by ISO code. Antarctica has no continent and was dropped.
4. **Missing values**
   - Population, GDP and energy: linear interpolation *inside* each country's time series, with no extrapolation.
   - Fuel breakdown (coal/oil/gas/…), missing for many small countries: set to 0. The remainder of total CO₂ is kept as *Other / unattributed* so totals still add up.
   - 125 rows with no CO₂ value were dropped.
   - GDP remains ~31% missing because the source has no GDP after 2022. GDP-based charts and models therefore use years ≤ 2022.
5. **Feature engineering:** GDP per capita, and each fuel's share of national CO₂ (%).
6. **Outliers:** the IQR rule on CO₂ per capita (upper fence ≈ 15.6 t) flags 471 rows, mostly Gulf petro-states, Australia and the US. These are **real values**, so they are **kept and flagged**, and the dashboard offers log scales to handle them.

All steps are reported live in the dashboard's **"Data & Preprocessing"** tab.

---

## 🧭 Dashboard overview

**Global interactive components (sidebar):** year-range slider · continent multiselect · map metric dropdown · log-scale checkbox · Top-N slider · filtered-data download.
**Chart-level components:** ranking radio + population checkbox (chart 3), country multiselect (chart 5), correlation-method radio (chart 6), *k* slider (chart 8), model dropdown + feature checkbox (chart 9).

| # | Visualization | Type | Interactivity | Question answered |
|---|---|---|---|---|
| 1 | Animated choropleth world map | **Advanced** (geo) | play/pause animation, hover tooltips, zoom/pan, metric dropdown | Where are emissions highest, and how did this change since 1990? |
| 2 | Sunburst (continent → country) | **Advanced** (hierarchical) | click to drill down, hover with % of parent | Who contributes to the global total? |
| 3 | Multi-line time series with range slider | Basic | range slider, unified hover, legend toggle | How did the top emitters evolve? |
| 4 | Sankey (fuel → continent) | **Advanced** (flow) | hover flows, drag nodes | Which fuels drive each continent's emissions? |
| 5 | Radar / polar fuel-mix chart | **Advanced** | choose countries, legend isolate, hover | What is each country's fuel-mix "fingerprint"? |
| 6 | Correlation heatmap | Basic | Pearson/Spearman switch, hover values | Which indicators move together? |
| 7 | Animated bubble chart (GDP vs CO₂, size = population) | **Advanced** | animation, hover, zoom, legend filter | Does getting richer mean emitting more? |
| 8 | **ML #1**: K-Means + parallel coordinates | **Advanced** (multivariate) | brush/drag axes to filter, *k* slider, silhouette curve | Which groups of countries share a similar profile? |
| 9 | **ML #2**: Regression, predicted vs actual + feature importance | ML / advanced | model dropdown, feature toggle, hover | Can wealth, energy and fuel mix predict CO₂ per capita? |

Chart 9 also includes a **feature-importance bar chart** and a **diverging error chart** (which countries the model over/under-estimates). The preprocessing tab contains a missing-value bar chart and an outlier box plot. A *How to use* panel and a **CSV download** of the filtered data are in the app.

### Machine learning
- **K-Means clustering** (scikit-learn) on standardized country profiles: log GDP/cap, log CO₂/cap, log energy/cap, coal/oil/gas shares. The silhouette score is shown for k = 2…8 to justify the choice of *k*.
- **Random Forest / Linear Regression** predicting log CO₂ per capita. The train/test split is **grouped by country** (`GroupShuffleSplit`), so the model is evaluated on countries it has never seen. This prevents leakage between neighbouring years. Random Forest gives R² ≈ 0.96 (≈ 0.88 without the energy feature). Model fits are cached (`st.cache_data`), so widget changes respond in about a second.

---

## 🚀 Setup & run

No installation needed to try it: the dashboard is deployed on Streamlit Community Cloud at https://co2-emissions-dashboardgit.streamlit.app/. To run it locally:

```bash
git clone https://github.com/omersalihotaci/co2-emissions-dashboard.git
cd co2-emissions-dashboard
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501 (on the first run Streamlit may ask for an e-mail address: just press Enter). Tested with Python 3.12, Streamlit 1.67, Plotly 6/7, scikit-learn 1.9.

### Project structure
```
├── app.py               # Streamlit application (all 9 visualizations + ML)
├── preprocess.py        # data loading, cleaning, outlier detection
├── data/
│   ├── owid-co2-data.csv    # raw dataset (OWID)
│   └── continents.csv       # country → continent mapping (OWID)
├── requirements.txt
├── REPORT.docx          # one-page project report
├── PRESENTATION.md      # speaking notes for the presentation
├── .streamlit/config.toml   # theme
└── README.md
```

---

## 👥 Team & contributions

| Team member | Responsibilities |
|---|---|
| **Ömer Salih Otacı** | Data preprocessing (`preprocess.py`), "Data & Preprocessing" tab; **Charts 1–3**: choropleth map, sunburst, top-emitters time series; sidebar filters |
| **Erdoğan Çalğan** | **Charts 4–6**: Sankey diagram, radar fuel-mix chart, correlation heatmap; visual consistency (shared colour palettes, template, labels) |
| **Abdullah Karabudak** | **Charts 7–9**: animated bubble chart, K-Means clustering + parallel coordinates, regression model + feature importance; README & report |

Each member presents their own three charts in the final presentation.

---

## 🔑 Key findings
1. Global CO₂ grew **~69% between 1990 and 2024**. **Asia now emits ~62%** of the total, and China alone ~53% of Asia.
2. **China overtook the US in 2006** and now emits more than twice as much. India rose from 0.58 Gt to 3.2 Gt and passed Russia.
3. **Coal is the largest single source**, and ~84% of coal CO₂ is emitted in Asia. Europe and the Americas emit mainly from oil and gas.
4. CO₂ per capita is strongly tied to **energy use (ρ ≈ 0.95)** and **wealth (ρ ≈ 0.90)**.
5. Several European countries show **decoupling**: e.g. the UK's GDP per capita rose ~46% (1990→2022) while its CO₂ per capita fell by more than half.
6. K-Means finds interpretable groups: low-income oil-only emitters, oil-based middle-income economies, gas-heavy emitters, and coal-heavy Asian economies.
