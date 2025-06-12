import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pearsonr, spearmanr, f_oneway, kruskal
import os

#--------------------------------------------------
# 1. Load the dataset
#--------------------------------------------------
file_path = r"C:\Users\ortho\OneDrive\Desktop\PendarLink AI\Registry\Dataset\Python_Analysis\data\dataset_26219_16.csv"
df = pd.read_csv(file_path)

#--------------------------------------------------
# 2. Define columns by type (matching your CSV exactly)
#   Age is explicitly categorical, not numeric.
#--------------------------------------------------
numeric_cols = [
    "Year of diagnosis",
    "Time from diagnosis to treatment in days recode",
    "Survival months"
]

categorical_cols = [
    "Age recode with less 1 year olds",
    "Sex",
    "Race recode White_Black_Other",
    "ICD_O_3 Hist_behav",
    "ICCC site recode extended 3rd edition_IARC 2017",
    "Site recode ICD_O_3_WHO 2008",
    "Primary Site _labeled",
    "Histologic Type ICD_O_3",
    "Reason no cancer_directed surgery",
    "Radiation recode",
    "Chemotherapy recode Yes_ No_Unknownunk",
    "SEER cause_specific death classification",
    "COD to site recode"
]

all_cols = numeric_cols + categorical_cols

# Ensure the DataFrame only has these columns (this step is optional if your CSV has exactly 16 columns)
df = df[all_cols]

#--------------------------------------------------
# 3. Helper functions
#--------------------------------------------------

def numeric_numeric_test(series1, series2, method="pearson"):
    """
    Returns correlation coefficient and p-value for two numeric series.
    You can switch 'method' to 'spearman' if needed.
    """
    # Drop rows with NaN in either column
    data = pd.concat([series1, series2], axis=1).dropna()
    if method == "pearson":
        corr, p_val = pearsonr(data.iloc[:,0], data.iloc[:,1])
        return ("Pearson", corr, p_val)
    else:
        corr, p_val = spearmanr(data.iloc[:,0], data.iloc[:,1])
        return ("Spearman", corr, p_val)

def categorical_categorical_test(series1, series2):
    """
    Returns the Chi-square statistic, p-value, degrees of freedom, and Cramér's V for two categorical series.
    """
    contingency = pd.crosstab(series1, series2)
    chi2, p_val, dof, expected = chi2_contingency(contingency)
    
    # Calculate Cramér's V
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt((chi2 / n) / min_dim)
    
    return chi2, p_val, dof, cramers_v

def numeric_categorical_test(num_series, cat_series, test_type="anova"):
    """
    For numeric vs. multi-level categorical:
      - ANOVA or Kruskal-Wallis
    If the categorical variable is strictly binary, 
    you might prefer T-test or Mann-Whitney.
    """
    df_temp = pd.concat([num_series, cat_series], axis=1).dropna()
    
    # Create groups for each category level
    groups = []
    for level in df_temp[cat_series.name].unique():
        groups.append(df_temp[df_temp[cat_series.name] == level][num_series.name])
    
    if test_type.lower() == "anova":
        stat, p_val = f_oneway(*groups)
        return ("ANOVA", stat, p_val)
    else:
        stat, p_val = kruskal(*groups)
        return ("Kruskal-Wallis", stat, p_val)

#--------------------------------------------------
# 4. Loop over all pairs of columns
#--------------------------------------------------
results = []

for i in range(len(all_cols)):
    for j in range(i+1, len(all_cols)):
        col1 = all_cols[i]
        col2 = all_cols[j]
        
        is_col1_numeric = col1 in numeric_cols
        is_col2_numeric = col2 in numeric_cols
        
        #--------------------------------------------------
        # Numeric vs. Numeric
        #--------------------------------------------------
        if is_col1_numeric and is_col2_numeric:
            test_name, stat_val, p_val = numeric_numeric_test(df[col1], df[col2], method="pearson")
            results.append({
                "Column1": col1,
                "Column2": col2,
                "Test": test_name,
                "Correlation_or_Statistic": stat_val,
                "p-value": p_val
            })
            
        #--------------------------------------------------
        # Categorical vs. Categorical
        #--------------------------------------------------
        elif (not is_col1_numeric) and (not is_col2_numeric):
            chi2, p_val, dof, cramers_v = categorical_categorical_test(df[col1], df[col2])
            results.append({
                "Column1": col1,
                "Column2": col2,
                "Test": "Chi-square",
                "Chi2": chi2,
                "p-value": p_val,
                "Degrees_of_Freedom": dof,
                "CramersV": cramers_v
            })
            
        #--------------------------------------------------
        # Numeric vs. Categorical
        #--------------------------------------------------
        else:
            if is_col1_numeric:
                num_col, cat_col = df[col1], df[col2]
            else:
                num_col, cat_col = df[col2], df[col1]
            
            test_type = "anova"  # or "kruskal" if data is non-normal
            test_name, stat_val, p_val = numeric_categorical_test(num_col, cat_col, test_type=test_type)
            results.append({
                "Column1": col1,
                "Column2": col2,
                "Test": test_name,
                "Statistic": stat_val,
                "p-value": p_val
            })

#--------------------------------------------------
# 5. Convert results to a DataFrame and save
#--------------------------------------------------
results_df = pd.DataFrame(results)

save_path = os.path.join(
    r"C:\Users\ortho\OneDrive\Desktop\PendarLink AI\Registry\Dataset\Python_Analysis\data",
    "pairwise_association_results.csv"
)
results_df.to_csv(save_path, index=False)

print("Analysis complete. Results saved to:", save_path)
