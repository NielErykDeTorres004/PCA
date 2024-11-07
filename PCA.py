# Import libraries & data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from scipy.stats import pearsonr
from numpy.linalg import eig
from sklearn.datasets import load_iris

# Load data
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add species information to the DataFrame
df['species'] = [data.target_names[i] for i in data.target]



# Define corrfunc function
def corrfunc(x, y, hue=None, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)  # Run the calculation on the x, y dataset given to the function
    ax = ax or plt.gca()   # Get the current plot axis the function is called for
    ax.annotate(f'r = {r:.2f}', xy=(.7, .1), xycoords=ax.transAxes)  # Add an annotation with the r-value



# Iris data histograms
axes = df.hist(bins=20, figsize=(10, 6), edgecolor='black')
plt.suptitle("Histograms of Iris Dataset Features")

# Turn off gridlines for each subplot
for ax in axes.flatten():
    ax.grid(False)

plt.savefig("iris_hist.svg", bbox_inches='tight')
plt.show()



# Iris data pairplot
g = sns.pairplot(df, 
                 hue='species',            # Color by species
                 kind='scatter',               # regression plot in main fields (scatter + regression line)
                 diag_kind='kde',          # smooth lines in the diagonal histogram fields
                 plot_kws={'alpha': 0.7},  # make scatter points semi-transparent
                 diag_kws={'color': 'tab:blue'})  # diagonal histograms using "DCU blue"

# Add the correlation annotation
g.map_offdiag(corrfunc)

# Add contour plot of lines of equal probability on the lower half of the pair plots
#g.map_lower(sns.kdeplot, levels=6, color='tab:green')
plt.savefig("iris_pairplot.svg", bbox_inches='tight')
plt.show()





# Standardization
means = df.iloc[:, :-1].mean()
std_devs = df.iloc[:, :-1].std()

# Standardize each value in the DataFrame (exclude species column)
df2 = (df.iloc[:, :-1] - means) / std_devs
df2.columns = [f"Standardised {col[:-4]}" for col in df.columns[:-1]]

# Add species column to standardized DataFrame
df2['species'] = df['species']







# Standardised data histograms
axes = df2.hist(bins=20, figsize=(10, 6), edgecolor='black')
plt.suptitle("Histograms of Iris Dataset Features")

# Turn off gridlines for each subplot
for ax in axes.flatten():
    ax.grid(False)

plt.savefig("standardised_hist.svg", bbox_inches='tight')
plt.show()


# Covariance Matrix
df_K = df2.iloc[:, :-1].cov()
K = df_K.values  # Convert the DataFrame covariance matrix to a NumPy array directly

# Eigenvalues and Eigenvectors with sorting
eigenvalue, eigenvector = eig(K)
idx = np.argsort(eigenvalue)[::-1]  # Sort eigenvalues in descending order
sorted_eigenvalues = eigenvalue[idx]
sorted_eigenvectors = eigenvector[:, idx]

print('Eigenvalues:', sorted_eigenvalues)
print('Eigenvectors:', sorted_eigenvectors)

# Magnitudes
magnitudes = [np.sqrt(sum(coef ** 2 for coef in vector)) for vector in sorted_eigenvectors.T]

# PCA Data Transformation
def reorient_data(df, eigenvectors):
    numpy_data = np.array(df)
    pca_features = np.dot(numpy_data, eigenvectors)
    return pd.DataFrame(pca_features)

pca_df = reorient_data(df2.iloc[:, :-1], sorted_eigenvectors)
pca_df.columns = ['PCA1', 'PCA2', 'PCA3', 'PCA4']
pca_df['species'] = df['species']  # Add species back to PCA DataFrame

# PCA Pairplot 
g4 = sns.pairplot(pca_df, 
                  hue='species', 
                  kind='scatter',  
                  diag_kind='kde', 
                  plot_kws={'alpha': 0.7}, 
                  diag_kws={'color': 'tab:blue'})

#g4.map_lower(sns.kdeplot, levels=6, color='tab:green')
plt.savefig("PC_pairplot.svg", bbox_inches='tight')
plt.show()
