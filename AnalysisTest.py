import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multicomp import pairwise_tukeyhsd

###Create Output Directory###
output_dir = "analysis_figures"
os.makedirs(output_dir, exist_ok=True)

###Load Data###
df = pd.read_csv("mixed_agent_results.csv")

###Create summary DataFrame###
agg_df = df.groupby(['turn', 'algorithm']).agg(
    final_score_mean=('final_score', 'mean'),
    final_score_sem=('final_score', stats.sem)
).reset_index()
agg_df['ci95'] = 1.96 * agg_df['final_score_sem']


###Descriptive Statistics###
print(df.groupby('algorithm')['final_score'].describe())

#Normality Check
print("\nShapiro-Wilk Normality Test:")
for algo in df['algorithm'].unique():
    stat, p = stats.shapiro(df[df['algorithm'] == algo]['final_score'])
    print(f"{algo}: W={stat:.3f}, p={p:.3f}")

###ANOVA and Tukey Post-Hoc Test###
algorithms = df['algorithm'].unique()
anova_result = stats.f_oneway(*[df[df['algorithm'] == a]['final_score'] for a in algorithms])
print("\nANOVA F-statistic:", anova_result.statistic, "p-value:", anova_result.pvalue)

tukey = pairwise_tukeyhsd(endog=df['final_score'], groups=df['algorithm'], alpha=0.05)
print(tukey)


###Learning Curve###
##Individual algorithm results##
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 5 plots in grid
axes = axes.flatten()

for i, alg in enumerate(df['algorithm'].unique()):
    ax = axes[i]
    subset = df[df['algorithm'] == alg]
    sns.lineplot(data=subset, x='turn', y='final_score', hue='run_id', legend=False, ax=ax)
    ax.set_title(alg)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Final Score")

fig.tight_layout()
plt.savefig(f"{output_dir}/individual_algorithm_score_curves.png", bbox_inches='tight')
plt.close()

##Average final score##
df_grouped = df.groupby(['algorithm', 'turn']).final_score.mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_grouped, x='turn', y='final_score', hue='algorithm')
plt.title("Average Final Score Over Time by Algorithm")
plt.xlabel("Turn")
plt.ylabel("Mean Final Score")
plt.legend(title="Algorithm")
plt.tight_layout()
plt.savefig(f"{output_dir}/mean_score_over_time.png", bbox_inches='tight')

###Regression: Score vs. Turn###
print("\nRegression Analysis (Performance vs Turn):")
for algo in algorithms:
    sub_df = agg_df[agg_df['algorithm'] == algo]
    X = sub_df[['turn']]
    y = sub_df['final_score_mean']
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    print(f"{algo} — Regression R²: {r2:.3f}, Slope: {model.coef_[0]:.3f}")

###Cohen's d Effect Sizes###
def cohens_d(x1, x2):
    return (x1.mean() - x2.mean()) / np.sqrt((x1.std()**2 + x2.std()**2) / 2)

print("\nCohen’s d Between Algorithm Pairs:")
for i in range(len(algorithms)):
    for j in range(i+1, len(algorithms)):
        d = cohens_d(df[df['algorithm'] == algorithms[i]]['final_score'],
                     df[df['algorithm'] == algorithms[j]]['final_score'])
        print(f"{algorithms[i]} vs {algorithms[j]}: d = {d:.3f}")

###KDE Plot###
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x="final_score", hue="algorithm", fill=True, common_norm=False, alpha=0.4)
plt.title("Score Distribution KDE")
plt.grid(True)
plt.savefig(f"{output_dir}/kde_score_distribution.png", bbox_inches='tight')
plt.close()


###Boxplot###
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="algorithm", y="final_score")
plt.title("Boxplot of Final Scores by Algorithm")
plt.grid(True)
plt.savefig(f"{output_dir}/boxplot_final_scores.png", bbox_inches='tight')
plt.close()

###Stripplot###
plt.figure(figsize=(10, 5))
sns.stripplot(data=df, x="algorithm", y="final_score", jitter=True, alpha=0.7)
plt.title("Final Scores per Run (Strip Plot)")
plt.grid(True)
plt.savefig(f"{output_dir}/stripplot_final_scores.png", bbox_inches='tight')
plt.close()

###Bar Plot###
means = df.groupby('algorithm')['final_score'].mean()
errors = df.groupby('algorithm')['final_score'].apply(lambda x: stats.sem(x))

plt.figure(figsize=(8, 5))
means.plot(kind='bar', yerr=errors, capsize=5, color='skyblue')
plt.ylabel("Mean Final Score")
plt.title("Mean Scores with 95% Confidence Intervals")
plt.grid(True)
plt.savefig(f"{output_dir}/barplot_mean_scores_confint.png", bbox_inches='tight')
plt.close()

###Coefficient of Variation (CV)###
cv_scores = df.groupby('algorithm')['final_score'].agg(['mean', 'std'])
cv_scores['CV'] = cv_scores['std'] / cv_scores['mean']
print("\nCoefficient of Variation:\n", cv_scores)
