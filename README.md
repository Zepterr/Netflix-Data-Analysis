# Netflix Movies & TV Shows — Data Analysis
### Introduction to Data Science · Spring 2026 · Undergraduate

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> **Discovering patterns & insights in streaming content using Python**

---

## Overview

Netflix hosts over **8,800 titles** spanning **127 countries**. This project applies 
foundational data science skills — cleaning, EDA, visualization, and machine learning — 
to uncover meaningful patterns in Netflix's global content catalog.

| Detail | Info |
|--------|------|
| 📚 Course | Introduction to Data Science |
| 📅 Semester | Spring 2026 |
| 📊 Dataset | Netflix Titles — Kaggle |
| 🗂️ Records | 8,794 titles · 12 features |
| 🌍 Countries | 127 represented |

---

## Team

| Name | Role |
|------|------|
| Syed Muhammad Zohaib | Data Wrangling & Cleaning |
| Laraib Fatima | EDA & Insights |
| Shahmir Khan | Data Visualization |
| Syed Jawad | Reporting & Documentation |

---

## Research Objectives

- Compare Movies vs TV Show proportions across catalog years
- Identify top content-producing countries using geographic data
- Analyze genre trends and how they evolved over time
- Examine content rating patterns by type
- Detect monthly/seasonal patterns in Netflix content additions
- Build a recommendation system to suggest similar titles

---

## Key Findings

| Insight | Finding |
|---------|---------|
| Content Split | 69.7% Movies · 30.3% TV Shows |
| Peak Growth | 2019 — highest ever at 2,016 titles added |
| COVID Impact | Sharp decline in 2020–2021 due to production shutdowns |
| Top Country | United States with 3,680 titles (41.8% of catalog) |
| #2 Country | India with 1,046 titles |
| Dominant Rating | TV-MA — Netflix targets adult audiences |
| Top Genre | International Movies (2,752 titles) |
| Peak Month | July — highest content additions across all years |
| Avg Movie Length | 100 minutes |
| Longest Title | Black Mirror: Bandersnatch at 312 minutes |

---

## Visualizations

| Chart | Description |
|-------|-------------|
| Chart 1 | Content Type Distribution — Donut Chart |
| Chart 2 | Netflix Growth Over Time — Line Chart |
| Chart 3 | Top 10 Content-Producing Countries — Bar Chart |
| Chart 4 | Top 10 Genres — Horizontal Bar Chart |
| Chart 5 | Monthly Content Additions — Heatmap |
| Chart 6 | Rating Distribution by Type — Stacked Bar |
| Chart 7 | Top 10 Directors — Horizontal Bar Chart |
| Chart 8 | Description Word Cloud |
| Chart 9 | Genre Trends Over Time — Line Chart |
| Chart 10 | Movie Duration Distribution — Histogram |

---

## Recommendation System

A content-based recommendation engine built using **TF-IDF vectorization** 
and **cosine similarity**, analyzing each title's genres, director, cast, 
and description.

**Matrix size:** 8,794 × 15,000 features

**Sample results:**
- *The Irishman* → GoodFellas, Raging Bull, Mean Streets ✅
- *Stranger Things* → Beyond Stranger Things (52% match) ✅
- *Inception* → Thrillers & Sci-Fi matches ✅

---

## Tools & Technologies

`Python 3.x` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `WordCloud` · `Jupyter Notebook` · `Google Colab` · `GitHub`

---

## Repository Structure
Netflix-Data-Analysis/
│
├── Netflix_Data_Analysis.ipynb   # Main analysis notebook
├── netflix_titles.csv            # Source dataset
├── chart1_content_type.png
├── chart2_growth_over_time.png
├── chart3_top_countries.png
├── chart4_top_genres.png
├── chart5_heatmap.png
├── chart6_ratings.png
├── chart7_directors.png
├── chart8_wordcloud.png
├── chart9_genre_trends.png
└── chart10_duration.png
---

*Introduction to Data Science · Spring 2026 · S.M. Zohaib · L. Fatima · S. Khan · S. Jawad*
