# --- GPU CHECK AND EARLY SETUP ---
import numpy as np  # Must be imported first
try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp  # Use cupy as a drop-in replacement for numpy
    print("GPU available via CuPy!")
except ImportError:
    GPU_AVAILABLE = False
    xp = np
    print("GPU not available; using CPU.")

# --- EARLY MONKEY-PATCH OF SCI-PY'S jensenshannon ---
import scipy.spatial.distance as ssd
from math import log2
def js_divergence(p, q, base=2):
    """Compute Jensen–Shannon Divergence between two probability distributions.
       p and q are expected to be xp.array objects (using CuPy if available)."""
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    jsd = 0.5 * xp.sum(p * xp.log(p / m)) + 0.5 * xp.sum(q * xp.log(q / m))
    if base == 2:
        jsd /= log2(xp.e if GPU_AVAILABLE else np.e)
    return jsd

# Override SciPy's jensenshannon with our custom version
ssd.jensenshannon = js_divergence

def js_distance(p, q, base=2):
    """Return the Jensen–Shannon distance (square root of divergence)."""
    from math import sqrt
    return sqrt(js_divergence(p, q, base=base))


# --- Standard Imports ---
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from math import sqrt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
import statsmodels.formula.api as smf
import json
import logging
from typing import Tuple, Dict

# -------------------------------
# Global File Paths & Output Directory
# -------------------------------
REAL_PATH  = r"C:\Users\ortho\OneDrive\Desktop\SDV\CTGAN\my_project\data\dataset_26219_16.csv"
SYNTH_PATH = r"C:\Users\ortho\OneDrive\Desktop\SDV\CTGAN\my_project\output\synthetic_data.csv"
OUTPUT_DIR = r"C:\Users\ortho\OneDrive\Desktop\SDV\CTGAN\my_project\output\evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Configuration
# -------------------------------
class Config:
    PATHS = {
        'real_data': REAL_PATH,
        'synthetic_data': SYNTH_PATH,
        'output_dir': OUTPUT_DIR
    }
    COLUMNS = {
        'numeric': [
            "Year of diagnosis", 
            "Time from diagnosis to treatment in days recode", 
            "Survival months"
        ],
        'categorical': [
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
    }
    MODEL_PARAMS = {
        'random_state': 42,
        'classification': {
            'target': "SEER cause_specific death classification",
            'features': [
                "Year of diagnosis", 
                "Time from diagnosis to treatment in days recode", 
                "Survival months"
            ]
        }
    }

# -------------------------------
# Data Loading & Preprocessing
# -------------------------------
class DataLoader:
    @staticmethod
    def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
        real = pd.read_csv(Config.PATHS['real_data'])
        synth = pd.read_csv(Config.PATHS['synthetic_data'])
        real.columns = real.columns.str.strip()
        synth.columns = synth.columns.str.strip()
        for col in Config.COLUMNS['numeric']:
            for df in [real, synth]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().mean() > 0.2:
                    logging.warning(f"High missingness in {col}")
        for col in Config.COLUMNS['categorical']:
            for df in [real, synth]:
                df[col] = df[col].astype('category').cat.as_ordered()
        return real, synth

# -------------------------------
# Evaluation Metrics
# -------------------------------
class EvaluationMetrics:
    @staticmethod
    def calculate_numeric_metrics(real: pd.Series, synth: pd.Series) -> Dict:
        real = real.dropna()
        synth = synth.dropna()
        if real.empty or synth.empty:
            logging.warning("Empty series encountered in numeric metrics.")
            return {}
        combined = pd.concat([real, synth])
        bins = np.histogram_bin_edges(combined, bins='auto')
        real_hist, _ = np.histogram(real, bins=bins)
        synth_hist, _ = np.histogram(synth, bins=bins)
        real_dist = xp.array(real_hist) / (real_hist.sum() + 1e-10)
        synth_dist = xp.array(synth_hist) / (synth_hist.sum() + 1e-10)
        js_val = js_divergence(real_dist, synth_dist, base=2)
        ks_stat, ks_p = ks_2samp(real, synth)
        X = xp.array(real).reshape(-1, 1)
        Y = xp.array(synth).reshape(-1, 1)
        mmd_val = mmd_rbf(X, Y)
        return {
            'KS Statistic': ks_stat,
            'p-value': ks_p,
            'Jensen-Shannon': float(js_val),
            'Wasserstein': wasserstein_distance(real, synth),
            'MMD^2': float(mmd_val)
        }
    
    @staticmethod
    def calculate_categorical_metrics(real: pd.Series, synth: pd.Series) -> Dict:
        real_probs = real.value_counts(normalize=True).sort_index()
        synth_probs = synth.value_counts(normalize=True).sort_index()
        all_idx = real_probs.index.union(synth_probs.index)
        real_probs = real_probs.reindex(all_idx, fill_value=0)
        synth_probs = synth_probs.reindex(all_idx, fill_value=0)
        js_val = js_divergence(real_probs.values, synth_probs.values, base=2)
        table, chi2, _ = contingency_table_analysis(real.name, real.name, pd.DataFrame({'real': real, 'synth': synth}))
        cv = cramers_v(pd.DataFrame({'real': real.value_counts(), 'synth': synth.value_counts()}))
        return {
            'Chi-Squared': chi2,
            'Cramér’s V': cv,
            'Jensen-Shannon': float(js_val)
        }

def mmd_rbf(X, Y, gamma=1.0):
    """Compute the Maximum Mean Discrepancy (MMD) with RBF kernel between two samples X and Y."""
    XX = xp.dot(X, X.T)
    XY = xp.dot(X, Y.T)
    YY = xp.dot(Y, Y.T)
    
    X_sqnorms = xp.diag(XX)
    Y_sqnorms = xp.diag(YY)
    
    K_XX = xp.exp(-gamma * (X_sqnorms[:, None] + X_sqnorms[None, :] - 2 * XX))
    K_XY = xp.exp(-gamma * (X_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * XY))
    K_YY = xp.exp(-gamma * (Y_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * YY))
    
    return xp.mean(K_XX) + xp.mean(K_YY) - 2 * xp.mean(K_XY)

# -------------------------------
# Univariate Analysis
# -------------------------------
class UnivariateAnalyzer:
    def __init__(self, real: pd.DataFrame, synth: pd.DataFrame):
        self.real = real
        self.synth = synth
        self.results = {}
        
    def analyze(self) -> Dict:
        self._analyze_numeric()
        self._analyze_categorical()
        return self.results
    
    def _analyze_numeric(self):
        for col in Config.COLUMNS['numeric']:
            self.results[col] = EvaluationMetrics.calculate_numeric_metrics(
                self.real[col], self.synth[col]
            )
    
    def _analyze_categorical(self):
        for col in Config.COLUMNS['categorical']:
            self.results[col] = EvaluationMetrics.calculate_categorical_metrics(
                self.real[col], self.synth[col]
            )

# -------------------------------
# Multivariate Analysis
# -------------------------------
class MultivariateAnalyzer:
    def __init__(self, real: pd.DataFrame, synth: pd.DataFrame):
        self.real = real
        self.synth = synth
        
    def analyze(self, output_dir: str):
        self._correlation_analysis(output_dir)
        self._dimensionality_reduction(output_dir)
        self._advanced_associations(output_dir)
    
    def _correlation_analysis(self, output_dir: str):
        for method in ['pearson', 'spearman']:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            sns.heatmap(
                self.real[Config.COLUMNS['numeric']].corr(method=method), 
                ax=axes[0], annot=True, cmap='coolwarm', vmin=-1, vmax=1
            )
            sns.heatmap(
                self.synth[Config.COLUMNS['numeric']].corr(method=method), 
                ax=axes[1], annot=True, cmap='coolwarm', vmin=-1, vmax=1
            )
            axes[0].set_title(f"Real Data ({method.title()} Correlation)")
            axes[1].set_title(f"Synthetic Data ({method.title()} Correlation)")
            plt.savefig(os.path.join(output_dir, f'{method}_correlation.png'), dpi=150)
            plt.close()
    
    def _dimensionality_reduction(self, output_dir: str):
        combined = pd.concat([
            self.real[Config.COLUMNS['numeric']].assign(Source='Real'),
            self.synth[Config.COLUMNS['numeric']].assign(Source='Synthetic')
        ])
        for reducer in [PCA(n_components=2), TSNE(n_components=2, perplexity=30, random_state=Config.MODEL_PARAMS['random_state'])]:
            reduced = reducer.fit_transform(combined.drop('Source', axis=1))
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                x=reduced[:, 0],
                y=reduced[:, 1],
                hue=combined['Source'],
                alpha=0.6,
                palette='Set1'
            )
            plt.title(f'{type(reducer).__name__} Visualization')
            plt.savefig(os.path.join(output_dir, f'{type(reducer).__name__}_plot.png'), dpi=150)
            plt.close()
    
    def _advanced_associations(self, output_dir: str):
        mi_val = mutual_info_discrete(self.real["Year of diagnosis"].dropna(), self.real["Survival months"].dropna())
        table, chi2_val, _ = contingency_table_analysis("Primary Site _labeled", "Reason no cancer_directed surgery", self.real)
        cv_val = cramers_v(table)
        adv_file = os.path.join(output_dir, "advanced_measures.txt")
        with open(adv_file, "w") as f:
            f.write(f"Mutual Information (Year vs Survival): {mi_val:.4f}\n")
            f.write(f"Cramér's V (Primary Site vs Reason no surgery): {cv_val:.4f}\n")

# -------------------------------
# Predictive Modeling
# -------------------------------
class PredictiveAnalyst:
    def __init__(self, real: pd.DataFrame, synth: pd.DataFrame):
        self.real = real
        self.synth = synth
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=Config.MODEL_PARAMS['random_state'])
        }
    
    def _prepare_target(self):
        target_col = Config.MODEL_PARAMS['classification']['target']
        self.le = LabelEncoder()
        combined = pd.concat([self.real[target_col], self.synth[target_col]])
        self.le.fit(combined.astype(str))
        for df in [self.real, self.synth]:
            df['target'] = self.le.transform(df[target_col].astype(str))
    
    def train_test_analysis(self, output_dir: str):
        self._prepare_target()
        self._tstr_analysis(output_dir)
        self._trts_analysis(output_dir)
    
    def _train_evaluate(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series, 
                        prefix: str, output_dir: str):
        results = []
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'F1 Macro': f1_score(y_test, y_pred, average='macro'),
                'ROC AUC': roc_auc_score(y_test, y_proba, multi_class='ovr'),
                'Classification Report': classification_report(y_test, y_pred)
            }
            results.append(metrics)
            disp = ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, display_labels=self.le.classes_, normalize='true'
            )
            disp.ax_.set_title(f'{name} Confusion Matrix ({prefix})')
            plt.savefig(os.path.join(output_dir, f'{prefix}_{name}_confusion.png'), dpi=150)
            plt.close()
        pd.DataFrame(results).to_csv(os.path.join(output_dir, f'{prefix}_metrics.csv'), index=False)
    
    def _tstr_analysis(self, output_dir: str):
        X_train = self.synth[Config.MODEL_PARAMS['classification']['features']]
        y_train = self.synth['target']
        X_test = self.real[Config.MODEL_PARAMS['classification']['features']]
        y_test = self.real['target']
        self._train_evaluate(X_train, y_train, X_test, y_test, 'TSTR', output_dir)
    
    def _trts_analysis(self, output_dir: str):
        X_train = self.real[Config.MODEL_PARAMS['classification']['features']]
        y_train = self.real['target']
        X_test = self.synth[Config.MODEL_PARAMS['classification']['features']]
        y_test = self.synth['target']
        self._train_evaluate(X_train, y_train, X_test, y_test, 'TRTS', output_dir)

# -------------------------------
# Regression Analysis
# -------------------------------
def regression_analysis(real_data, synth_data, output_dir):
    outcome_col = "SEER cause_specific death classification"
    feature_cols = ["Year of diagnosis", "Time from diagnosis to treatment in days recode", "Survival months"]
    def binarize_outcome(val):
        return 1 if "Dead" in val else 0
    real_data["target"] = real_data[outcome_col].astype(str).apply(binarize_outcome)
    synth_data["target"] = synth_data[outcome_col].astype(str).apply(binarize_outcome)
    real_data = real_data.dropna(subset=feature_cols + ["target"])
    synth_data = synth_data.dropna(subset=feature_cols + ["target"])
    real_data = real_data.rename(columns={"Year of diagnosis": "Year_of_diagnosis",
                                           "Time from diagnosis to treatment in days recode": "Time_to_tx",
                                           "Survival months": "Survival_months"})
    synth_data = synth_data.rename(columns={"Year of diagnosis": "Year_of_diagnosis",
                                             "Time from diagnosis to treatment in days recode": "Time_to_tx",
                                             "Survival months": "Survival_months"})
    formula = "target ~ Year_of_diagnosis + Time_to_tx + Survival_months"
    model_real = smf.logit(formula=formula, data=real_data).fit(disp=False)
    model_synth = smf.logit(formula=formula, data=synth_data).fit(disp=False)
    real_summary_df = pd.DataFrame({"Coefficient": model_real.params,
                                    "CI_lower": model_real.conf_int()[0],
                                    "CI_upper": model_real.conf_int()[1],
                                    "p_value": model_real.pvalues})
    real_summary_df.index.name = "Term"
    synth_summary_df = pd.DataFrame({"Coefficient": model_synth.params,
                                     "CI_lower": model_synth.conf_int()[0],
                                     "CI_upper": model_synth.conf_int()[1],
                                     "p_value": model_synth.pvalues})
    synth_summary_df.index.name = "Term"
    compare_df = real_summary_df.add_suffix("_Real").join(synth_summary_df.add_suffix("_Synth"), how="outer")
    compare_df.to_csv(os.path.join(output_dir, "logit_compare.csv"))
    real_summary_df.to_csv(os.path.join(output_dir, "logit_real_summary.csv"))
    synth_summary_df.to_csv(os.path.join(output_dir, "logit_synth_summary.csv"))
    print("Regression analysis complete. Results saved in", output_dir)

# -------------------------------
# Benchmarking Analysis (Advanced Measures)
# -------------------------------
def benchmarking_analysis(real_data, synth_data, output_dir):
    site_col = "Site recode ICD_O_3_WHO 2008"
    histo_col = "Histologic Type ICD_O_3"
    mi_val = mutual_info_discrete(real_data[site_col].dropna(), real_data[histo_col].dropna())
    primary_site = "Primary Site _labeled"
    reason_no_surgery = "Reason no cancer_directed surgery"
    table, chi2_val, p_val = contingency_table_analysis(primary_site, reason_no_surgery, real_data)
    cramer_val = cramers_v(table)
    adv_file = os.path.join(output_dir, "advanced_measures.txt")
    with open(adv_file, "w") as f:
        f.write(f"Mutual Information ({site_col} vs {histo_col}): {mi_val:.4f}\n")
        f.write(f"Cramér's V ({primary_site} vs {reason_no_surgery}): {cramer_val:.4f}\n")
    print("Benchmarking analysis complete. Advanced measures saved in", output_dir)

# -------------------------------
# Main Function: Run All Analyses
# -------------------------------
def main():
    os.makedirs(Config.PATHS['output_dir'], exist_ok=True)
    logging.basicConfig(filename=os.path.join(Config.PATHS['output_dir'], 'evaluation.log'),
                        level=logging.INFO)
    
    real_data, synth_data = DataLoader.load_data()
    
    print("Running Univariate Analysis...")
    uni_analyzer = UnivariateAnalyzer(real_data, synth_data)
    uni_results = uni_analyzer.analyze()
    with open(os.path.join(Config.PATHS['output_dir'], 'univariate.json'), 'w') as f:
        json.dump(uni_results, f, indent=2)
    
    print("Running Multivariate Analysis...")
    multi_analyzer = MultivariateAnalyzer(real_data, synth_data)
    multi_analyzer.analyze(Config.PATHS['output_dir'])
    
    print("Running Predictive Modeling (TSTR)...")
    predictor = PredictiveAnalyst(real_data, synth_data)
    predictor.train_test_analysis(Config.PATHS['output_dir'])
    
    print("Running Regression Analysis...")
    regression_analysis(real_data, synth_data, Config.PATHS['output_dir'])
    
    print("Running Benchmarking Analysis...")
    benchmarking_analysis(real_data, synth_data, Config.PATHS['output_dir'])
    
    print("All analyses complete. Check the output folder for consolidated results.")

if __name__ == "__main__":
    main()
