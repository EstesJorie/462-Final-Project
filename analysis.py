# --------------------------------------
# 🛠️ Imports
# --------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm

# --------------------------------------
# 🗂️ Create Output Directory
# --------------------------------------
output_dir = "analysis_figures"
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------
# 🎲 Simulated Test Data (replace with your real data)
# --------------------------------------
np.random.seed(42)

df = pd.read_csv("logs/sim_mixed_agent_scores.csv")

# --------------------------------------
# 📊 1. Descriptive Statistics
# --------------------------------------
print(df.groupby('algorithm')['final_score'].describe())

# --------------------------------------
# 🧪 2. Normality Check
# --------------------------------------
print("\nShapiro-Wilk Normality Test:")
for algo in df['algorithm'].unique():
    stat, p = stats.shapiro(df[df['algorithm'] == algo]['final_score'])
    print(f"{algo}: W={stat:.3f}, p={p:.3f}")

# --------------------------------------
# 🔬 3. ANOVA and Tukey Post-Hoc Test
# --------------------------------------
anova_result = stats.f_oneway(*[df[df['algorithm'] == a]['final_score'] for a in algorithms])
print("\nANOVA F-statistic:", anova_result.statistic, "p-value:", anova_result.pvalue)

tukey = pairwise_tukeyhsd(endog=df['final_score'], groups=df['algorithm'], alpha=0.05)
print(tukey)

# --------------------------------------
# 📈 4. Learning Curves: Final Score over Turns
# --------------------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="turn", y="final_score", hue="algorithm", ci='sd')
plt.title("Final Score over Turns")
plt.xlabel("Turn")
plt.ylabel("Final Score")
plt.grid(True)
plt.savefig(f"{output_dir}/final_score_over_turns.png", bbox_inches='tight')
plt.close()

# --------------------------------------
# 📈 5. Score over Training Episodes
# --------------------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="episode", y="final_score", hue="algorithm", ci="sd")
plt.title("Final Score vs. Training Episodes")
plt.xlabel("Training Episode")
plt.ylabel("Final Score")
plt.grid(True)
plt.savefig(f"{output_dir}/final_score_over_episodes.png", bbox_inches='tight')
plt.close()

# --------------------------------------
# 📈 6. Regression: Score vs. Training Episode
# --------------------------------------
print("\nRegression Analysis (Performance vs Training Episodes):")
for algo in algorithms:
    sub_df = df[df['algorithm'] == algo]
    X = sub_df[['episode']]
    y = sub_df['final_score']
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    print(f"{algo} — Regression R²: {r2:.3f}, Slope: {model.coef_[0]:.3f}")

# --------------------------------------
# 📈 7. Cohen's d Effect Sizes
# --------------------------------------
def cohens_d(x1, x2):
    return (x1.mean() - x2.mean()) / np.sqrt((x1.std()**2 + x2.std()**2) / 2)

print("\nCohen’s d Between Algorithm Pairs:")
for i in range(len(algorithms)):
    for j in range(i+1, len(algorithms)):
        d = cohens_d(df[df['algorithm'] == algorithms[i]]['final_score'],
                     df[df['algorithm'] == algorithms[j]]['final_score'])
        print(f"{algorithms[i]} vs {algorithms[j]}: d = {d:.3f}")

# --------------------------------------
# 📈 8. KDE Plot: Final Score Distribution
# --------------------------------------
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x="final_score", hue="algorithm", fill=True, common_norm=False, alpha=0.4)
plt.title("Score Distribution KDE")
plt.grid(True)
plt.savefig(f"{output_dir}/kde_score_distribution.png", bbox_inches='tight')
plt.close()

# --------------------------------------
# 📈 9. Boxplot: Final Score Distribution
# --------------------------------------
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="algorithm", y="final_score")
plt.title("Boxplot of Final Scores by Algorithm")
plt.grid(True)
plt.savefig(f"{output_dir}/boxplot_final_scores.png", bbox_inches='tight')
plt.close()

# --------------------------------------
# 📈 10. Stripplot: Final Score per Run
# --------------------------------------
plt.figure(figsize=(10, 5))
sns.stripplot(data=df, x="algorithm", y="final_score", jitter=True, alpha=0.7)
plt.title("Final Scores per Run (Strip Plot)")
plt.grid(True)
plt.savefig(f"{output_dir}/stripplot_final_scores.png", bbox_inches='tight')
plt.close()

# --------------------------------------
# 📈 11. Bar Plot with 95% Confidence Intervals
# --------------------------------------
means = df.groupby('algorithm')['final_score'].mean()
errors = df.groupby('algorithm')['final_score'].apply(lambda x: stats.sem(x))

plt.figure(figsize=(8, 5))
means.plot(kind='bar', yerr=errors, capsize=5, color='skyblue')
plt.ylabel("Mean Final Score")
plt.title("Mean Scores with 95% Confidence Intervals")
plt.grid(True)
plt.savefig(f"{output_dir}/barplot_mean_scores_confint.png", bbox_inches='tight')
plt.close()

# --------------------------------------
# 📊 12. Coefficient of Variation (CV)
# --------------------------------------
cv_scores = df.groupby('algorithm')['final_score'].agg(['mean', 'std'])
cv_scores['CV'] = cv_scores['std'] / cv_scores['mean']
print("\nCoefficient of Variation:\n", cv_scores)

# --------------------------------------
# 🎯 13. (Optional) Action Frequencies
# --------------------------------------
if 'action' in df.columns:
    action_counts = df.groupby(['turn', 'algorithm', 'action']).size().reset_index(name='count')
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=action_counts, x='turn', y='count', hue='action', style='algorithm')
    plt.title("Action Frequencies Over Turns")
    plt.savefig(f"{output_dir}/action_frequencies_over_turns.png", bbox_inches='tight')
    plt.close()
