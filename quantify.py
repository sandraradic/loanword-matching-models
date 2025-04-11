import Levenshtein
import pandas as pd
import matplotlib.pyplot as plt

# Sample data: English words and their Serbian adaptations
data = {
    'english_word': ['like', 'hate', 'online', 'tweet', 'facebook', 'chat', 'show', 'shopping', 'sorry', 'smiley'],
    'serbian_word': ['lajk', 'hejt', 'na lajni', 'tvit', 'fejs', 'cet', 'sou', 'soping', 'sori', 'smajli']
}

df = pd.DataFrame(data)

# Function to calculate distances and similarities
def calculate_distances(row):
    en_word = row['english_word']
    sr_word = row['serbian_word']
    lev_distance = Levenshtein.distance(en_word, sr_word)
    jw_similarity = Levenshtein.jaro_winkler(en_word, sr_word)
    return pd.Series({'levenshtein_distance': lev_distance, 'jaro_winkler_similarity': jw_similarity})

# Apply the function to each row
df[['levenshtein_distance', 'jaro_winkler_similarity']] = df.apply(calculate_distances, axis=1)

print(df)

# FUZZY SCORE
from rapidfuzz import fuzz

def get_fuzzy_score(token, canonical):
    # Returns a similarity score (0-100)
    return fuzz.ratio(token, canonical)

score = get_fuzzy_score("like", "lajk")
print("Fuzzy score:", score)


# Box Plot data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# DataFrame
df_metrics = pd.DataFrame({
    'english_word': ['like', 'hate', 'online', 'tweet', 'facebook', 'chat', 'show', 'shopping', 'sorry', 'smiley'],
    'serbian_word': ['lajk', 'hejt', 'na lajni', 'tvit', 'fejs', 'cet', 'sou', 'soping', 'sori', 'smajli'],
    'levenshtein_distance': [3, 3, 6, 3, 6, 2, 2, 2, 2, 4], #divided by 100 to normalize with jw-sim
    'jaro_winkler_similarity': [0.666667, 0.666667, 0.625000, .633333, 0.583333, 0.750000, 0.750000, 0.925000, 0.848333, 0.666667],
    'classification': ["fuzzy", "fuzzy", "fuzzy", "fuzzy", "fuzzy", "fuzzy", "fuzzy", "fuzzy", "fuzzy", "fuzzy"]  # "pure" or "fuzzy"
})

# Normalizing the Levenshtein similarity by dividing by 100 to get values between 0 and 1.
df_metrics['norm_lev'] = df_metrics['levenshtein_distance'] / 100.0
df_metrics["adj_lev"] = 1 - df_metrics['norm_lev']
print(df_metrics["adj_lev"])

# For Jaro-Winkler, assume scores already on a 0-1 scale.
df_metrics['norm_jw'] = df_metrics['jaro_winkler_similarity']

# Melt the DataFrame so both normalized metrics are in one column for side-by-side comparison.
df_melted = df_metrics.melt(
    id_vars=['serbian_word', 'classification'], 
    value_vars=['adj_lev', 'norm_jw'], 
    var_name='Metric', 
    value_name='Score'
)

plt.figure(figsize=(8, 6))
sns.boxplot(x='Metric', y='Score', data=df_melted)
plt.title("Normalized Similarity Metrics for Fuzzy Matches")
plt.ylabel("Normalized Similarity Score (0-1)")
plt.xlabel("Metric")
plt.show()

# Statistical Significance testing
import numpy as np
from scipy import stats

# Example data for two groups of fuzzy tokens (normalized scores between 0 and 1)
# Group 1: Normalized Levenshtein similarity scores (0-1 scale)
# Group 2: Jaro-Winkler similarity scores (0-1 scale)
group_1 = np.array([0.97, 0.97, 0.94, 0.97, 0.94, 0.98, 0.98, 0.98, 0.98, 0.96])  # More English-like
group_2 = np.array([0.666667, 0.666667, 0.625000, 0.633333, 0.583333, 0.750000, 0.750000, 0.925000, 0.848333, 0.666667])  # More Serbian-like

# Run independent two-sample t test on Similarity Calculations
t_stat, p_val = stats.ttest_ind(group_1, group_2, equal_var=False)  

print("T-statistic:", t_stat)
print("P-value:", p_val)

# p-value interpretation
if p_val < 0.05:
    print("There is a statistically significant difference between the two groups.")
else:
    print("There is no statistically significant difference between the two groups.")


# Run paired t test
import numpy as np
from scipy.stats import ttest_rel

# English vs. fuzzy frequencies for top 10 words:
english_freqs = np.array([6,9,8,20,1,1,0,1,0,2])
fuzzy_freqs   = np.array([134,50,29,23,12,12,5,4,3,3])

# Paired t-test
t_stat, p_val = ttest_rel(english_freqs, fuzzy_freqs)
print("T-statistic:", t_stat)
print("p-value:", p_val)

if p_val < 0.05:
    print("Statistically significant difference.")
else:
    print("No statistically significant difference.")




