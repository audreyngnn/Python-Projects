# Airbnb Melbourne Price Prediction

## 📌 Project Overview
This project focuses on building a **machine learning model to predict the price of Airbnb listings in Melbourne** based on property and host features.
The model was trained on historical data and applied to unseen listings, with the aim of achieving high predictive accuracy.

### Business Relevance (Stakeholders)
- **Hosts**: Set competitive, data-driven prices and avoid revenue loss from underpricing or booking loss from overpricing
- **Guests**: Compare predicted fair prices against listings to make better booking decisions
- **Platforms**: Enhance pricing systems with more transparent, accurate, and explainable algorithms

<br>

## 📂 Dataset
The project was built on structured Airbnb listing data provided in two files:
- **train.csv**: 7,000 rows × 61 columns (includes target variable: `price`)
- **test.csv**: 3,000 rows × 60 columns (excludes target variable: `price`)

<br>

## 🔎 Project Workflow

### 1. Exploratory Data Analysis (EDA)
`Part 1_EDA.ipynb`

- Explored property, host, and location features to understand data distribution
- Identified outliers and highlighted correlations between variables and price
- Built early insights into which factors (e.g., location, amenities, host behavior) drive price differences

<br>

### 2. Data Cleaning and Engineering 
`Part 2_Data_Engineering.ipynb`

**The Pipeline:**
```python
def preprocess(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_features(df)                    # Standardize formats (e.g., "$150" → 150.0, "95%" → 0.95)
    df = create_new_features(df)               # Extract business logic (host_tenure_years, price_per_bed, is_superhost)
    df = add_extra_features(df)                # Domain knowledge (amenity_count, cancellation_strictness, has_instant_book)
    df = add_distance_features(df)             # Location context (distance to CBD, MCG, St Kilda Beach)
    df = add_interaction_features(df)          # Non-linear relationships (bedrooms × bathrooms, price_per_review)
    df = impute_missing(df, ref_df)            # Handle gaps using training set statistics (prevents data leakage)
    df = encode_amenities(df, ref_df)          # Convert amenity lists to binary flags; group rare → "Other"
    df = encode_neighbourhoods(df, ref_df)     # Simplify 300+ areas into top 20 + "Other"
    df = encode_other_features(df, ref_df)     # Encode property_type, room_type, cancellation_policy
    df = apply_feature_transformations(df)     # Log-transform skewed features to normalize distributions
    return df
```

**Why This Matters:**

| Principle | Impact |
|-----------|--------|
| **Consistency** | Train and test processed identically using `ref_df` to prevent data leakage |
| **Feature Engineering** | Translates domain knowledge (host experience, location value) into predictive signals |
| **Noise Reduction** | Groups rare categories to prevent overfitting on outliers |
| **Scalability** | Modular functions make the pipeline reusable for other datasets |

A comprehensive data pipeline ensures the model learns from **reliable, consistent, and business-relevant variables**, which has a greater impact on accuracy than algorithm choice alone.

<br>

### 3. Predictive Modeling
`Part 3_Predictives.ipynb`
- Trained four predictive models: **Ridge, Lasso, SVR, and LightGBM**
- Built a **stacked ensemble** combining these models to reduce error and capture complementary strengths
- Used **Optuna** for hyperparameter tuning, balancing accuracy with generalization

<br>

## 🏆 Results

- **Final Ranking**: Top 3 out of 106 competitors
- **Private Leaderboard Score**: **110.299** (predictions within ~110 AUD of actual prices on average)
- **Result Recorded**: 6 June 2025, 1:57 PM (AEST)
- **Leaderboard Reference**: [View on Kaggle](https://www.kaggle.com/competitions/asba-predictive-analytics-competition/leaderboard)
- **Professor Letter**: [View Here](https://github.com/audreyngnn/Python-Projects/blob/main/Professor%20Recognition%20Letter.pdf)

![](Final_Results.PNG)

<br>

## 💡 Key Takeaways

- **Data quality is the driver of performance**: Well-structured inputs had a bigger impact than complex models
- **Generalizable pipeline**: The feature engineering and cleaning framework can be applied to other pricing or recommendation problems
- **Exposure to ML**: Built and tuned predictive models to demonstrate how clean data enables stronger outcomes

<br>
