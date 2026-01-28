import numpy as np
from sklearn.preprocessing import StandardScaler

# Simple Hotel Dataset (only integers)
X = np.array([
    [2000, 4, 5, 50],
    [3000, 5, 2, 80],
    [1500, 3, 8, 40],
    [4000, 5, 1, 100],
    [2500, 4, 4, 60]
])

print("Original Dataset:\n", X)

# Step 3: Standardization
X_std = StandardScaler().fit_transform(X)
print("\nStandardized Data:\n", X_std)

# Step 4: Covariance Matrix
cov_matrix = np.cov(X_std.T)
print("\nCovariance Matrix:\n", cov_matrix)

# Step 5: Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

# Step 6: Sort eigenvalues
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nSorted Eigenvalues:\n", eigenvalues)
print("\nSorted Eigenvectors:\n", eigenvectors)

# Step 7: Select top k components
k = 1
principal_components = eigenvectors[:, :k]
print("\nPrincipal Component Matrix:\n", principal_components)

# Step 9: Reduced Dataset
X_reduced = np.dot(X_std, principal_components)
print("\nReduced Dataset (After PCA):\n", X_reduced)

# Step 10: Variance retained
variance_ratio = eigenvalues / np.sum(eigenvalues)
print("\nVariance Percentage:\n", variance_ratio * 100)




