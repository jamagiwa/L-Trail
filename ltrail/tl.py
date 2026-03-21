import numpy as np
from scipy.stats import skew
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors


#-----------------------------------------------

def _calc_l_moment_vector(data: np.ndarray) -> np.ndarray:
    """
    Estimate the transition direction of a cell population by calculating the L-moment 
    (specifically L-skewness, tau3) along each dimension of the input data.

    This function calculates Probability Weighted Moments (PWM) and linearly transforms 
    them into L-moments. The resulting L-skewness indicates the geometric asymmetry 
    ("comet tail") of the distribution, which is used to define the L-Trail vector.

    Parameters
    ----------
    data : np.ndarray
        Input data array representing the coordinates of a cell population in a 
        multi-dimensional space (e.g., PCA coordinates).
        Shape should be (n_cells, n_dimensions).

    Returns
    -------
    vec : np.ndarray
        The estimated directional vector (L-Flow) representing the transition direction.
        Shape is (n_dimensions,).
    """
    n_cells = data.shape[0]
    n_dims = data.shape[1]

    # Return a zero vector if the number of cells is insufficient for robust calculation
    if n_cells < 30:
        return np.zeros(n_dims)

    # 1. Order Statistics
    # Sort the data along each dimension: X(1) <= X(2) <= ... <= X(n)
    sorted_data = np.sort(data, axis=0)

    # ----------------------------------------------------------------------
    # 2. Probability Weighted Moments (PWM) Calculation
    # Formula: b_r = (1/n) * sum( x_i * weight_i )
    # ----------------------------------------------------------------------
    idx = np.arange(1, n_cells + 1)

    # b0: Equivalent to the standard mean
    b0 = np.mean(sorted_data, axis=0)

    # b1: Weighting by (i - 1) / (n - 1)
    w1 = (idx - 1) / (n_cells - 1)
    b1 = np.mean(sorted_data * w1[:, None], axis=0)

    # b2: Weighting by (i - 1)(i - 2) / ((n - 1)(n - 2))
    w2 = ((idx - 1) * (idx - 2)) / ((n_cells - 1) * (n_cells - 2))
    b2 = np.mean(sorted_data * w2[:, None], axis=0)

    # ----------------------------------------------------------------------
    # 3. Derivation of L-moments from PWMs
    # ----------------------------------------------------------------------
    # lamda2 (L-scale): Represents dispersion
    lamda2 = 2 * b1 - b0

    # lamda3 (L-3rd moment): Represents asymmetry / skewness
    lamda3 = 6 * b2 - 6 * b1 + b0


    # ----------------------------------------------------------------------
    # 4. Definition of L-Flow (Directional Vector)
    # ----------------------------------------------------------------------
    # The vector direction is determined by the negative skewness (-tau3),
    vec = -lamda3

    return vec

#-----------------------------------------------
def _calc_high_dim_vector(
    data: np.ndarray,
    method: str = 'lmoment',
    scale_skew_std: bool = True
) -> tuple:
    """
    Calculate the centroid and directional vector for a given cluster in high-dimensional space.

    Parameters
    ----------
    data : np.ndarray
        High-dimensional data array for a specific cluster (e.g., PCA coordinates).
        Shape should be (n_cells, n_dimensions).
    method : str, optional (default: 'lmoment')
        Method used to calculate the directional vector representing the trajectory.
        Available options are:
        - 'lmoment': Vector based on L-moments (recommended).
        - 'pearson': Vector based on Pearson's second coefficient of skewness.
        - 'skew': Vector based on standard Fisher-Pearson coefficient of skewness.
    scale_skew_std : bool, optional (default: True)
        Whether to scale the standard skewness vector by the standard deviation.
        Only applicable when `method='skew'`.

    Returns
    -------
    mean_pos : np.ndarray
        The centroid (mean position) of the cluster.
    vec : np.ndarray
        The calculated directional vector indicating the trajectory of the cluster.
    """
    mean_pos = np.mean(data, axis=0)

    if method == 'pearson':
        # Utilizing the difference between mean and median
        # Based on Pearson's second coefficient of skewness
        median_pos = np.median(data, axis=0)
        vec = 3 * (median_pos - mean_pos)

    elif method == 'skew':
        # Standard Fisher-Pearson coefficient of skewness
        sk = skew(data, axis=0)
        if scale_skew_std:
            std = np.std(data, axis=0)
            vec = -sk * std
        else:
            vec = -sk

    elif method == 'lmoment':
        # L-moment based vector calculation
        vec = _calc_l_moment_vector(data)
        
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'lmoment', 'pearson', or 'skew'.")

    return mean_pos, vec

#-----------------------------------------------
def _test_significance_high_dim(
    subset: np.ndarray,
    observed_magnitude: float,
    method: str,
    n_boot: int = 1000, 
    random_state; int = 34
) -> float:
    """
    Perform a permutation test to calculate the statistical significance (p-value) of the directional vector.

    This function generates a null distribution by centering the data and randomly 
    flipping the signs of the coordinates, testing the null hypothesis that the 
    observed directional magnitude is due to chance.

    Parameters
    ----------
    subset : np.ndarray
        High-dimensional data array for a specific cluster (e.g., PCA coordinates).
        Shape should be (n_cells, n_dimensions).
    observed_magnitude : float
        The magnitude (Euclidean length) of the observed directional vector.
    method : str
        Method used to calculate the directional vector (e.g., 'lmoment', 'pearson', 'skew').
        This is passed internally to `_calc_high_dim_vector`.
    n_boot : int, optional (default: 1000)
        Number of permutations (bootstrap iterations) to perform for generating 
        the null distribution.

    Returns
    -------
    p_val : float
        The calculated empirical p-value representing the statistical significance 
        of the directional vector.
    """
    # Centering the data to set up the null hypothesis
    centered = subset - np.mean(subset, axis=0)
    n_cells = centered.shape[0]
    null_mags = []

    for _ in range(n_boot):
        # Generate data with randomly flipped signs (1 or -1)
        signs = np.random.choice([-1, 1], size=(n_cells, 1))
        shuffled = centered * signs
        
        try:
            # Calculate the vector for the shuffled null data
            _, vec_null = _calc_high_dim_vector(shuffled, method=method)
            mag_null = np.sqrt(np.sum(vec_null**2))
            null_mags.append(mag_null)
        except Exception:
            # Append 0.0 if calculation fails for extreme random cases
            null_mags.append(0.0)

    null_mags = np.array(null_mags)

    # Calculate the empirical p-value
    p_val = (np.sum(null_mags >= observed_magnitude) + 1) / (n_boot + 1)
    
    return p_val

#-----------------------------------------------
def calc_velocity_ltrail_similarity(
    adata,
    groupby: str,
    use_rep: str = 'X_pca',
    vel_rep: str = 'velocity_pca',
    method: str = 'lmoment',
    n_pca: int = 30,
    min_cells: int = 20
) -> pd.DataFrame:
    """
    Calculate the cosine similarity between RNA velocity vectors and L-Trail vectors for each cluster.

    This function computes the mean RNA velocity vector for a given cluster and 
    compares it with the L-Trail directional vector calculated in the same high-dimensional space 
    using cosine similarity.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the single-cell dataset.
    groupby : str
        Key in `adata.obs` defining the clusters for vector calculation.
    use_rep : str, optional (default: 'X_pca')
        Key in `adata.obsm` for the high-dimensional space used for calculation.
    vel_rep : str, optional (default: 'velocity_pca')
        Key in `adata.obsm` containing the RNA velocity vectors projected in the high-dimensional space.
    method : str, optional (default: 'lmoment')
        Method used for calculating the L-Trail directional vector ('lmoment', 'pearson', 'skew').
    n_pca : int, optional (default: 30)
        Number of principal components to use from the high-dimensional space.
    min_cells : int, optional (default: 20)
        Minimum number of cells required in a cluster to calculate its vector.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the cluster name, cosine similarity score, and calculation status (Note),
        sorted in descending order of cosine similarity.
    """
    if use_rep not in adata.obsm.keys():
        raise ValueError(f"'{use_rep}' not found in adata.obsm.")
    if vel_rep not in adata.obsm.keys():
        raise ValueError(f"'{vel_rep}' not found in adata.obsm. Did you run scVelo/Velocity first?")

    X_high = adata.obsm[use_rep][:, :n_pca]
    V_high = adata.obsm[vel_rep][:, :n_pca]  # Velocity vectors
    
    groups = adata.obs[groupby]
    unique_groups = groups.unique()

    results = []

    for group in unique_groups:
        mask = (groups == group)
        subset_high = X_high[mask]
        subset_vel = V_high[mask]

        if len(subset_high) < min_cells:
            continue

        try:
            # 1. Calculate L-Trail vector
            _, vec_ltrail = _calc_high_dim_vector(subset_high, method=method)

            # 2. Calculate the mean RNA velocity vector for the cluster
            vec_vel = np.mean(subset_vel, axis=0)

            # Avoid zero/noise vectors
            mag_ltrail = np.linalg.norm(vec_ltrail)
            mag_vel = np.linalg.norm(vec_vel)

            if mag_ltrail < 1e-6 or mag_vel < 1e-6:
                results.append({
                    'Cluster': group, 
                    'Cos_Similarity': np.nan, 
                    'Note': 'zero vector'
                })
                continue

            # 3. Calculate Cosine Similarity
            # Note: scipy.spatial.distance.cosine returns the cosine distance (1 - similarity)
            sim = 1.0 - cosine(vec_vel, vec_ltrail)

            results.append({
                'Cluster': group,
                'Cos_Similarity': sim,
                'Note': 'Success'
            })

        except Exception as e:
            results.append({
                'Cluster': group, 
                'Cos_Similarity': np.nan, 
                'Note': f'Error: {e}'
            })

    # Compile results into a DataFrame and sort
    df_results = pd.DataFrame(results).sort_values(by='Cos_Similarity', ascending=False)
    
    return df_results


#-----------------------------------------------

def calc_grid_similarity(
    adata,
    use_rep: str = 'X_pca',
    vel_rep: str = 'velocity_pca',
    n_pca: int = 30,
    method: str = 'lmoment',
    grid_size: int = 15,
    min_cells: int = 20
) -> pd.DataFrame:
    """
    Calculate the cosine similarity between RNA velocity and L-Trail vectors across a spatial grid.

    This function divides the top two dimensions of the specified representation space 
    into a uniform grid, and calculates the cosine similarity between the mean RNA velocity 
    and the L-Trail directional vector for the cells within each grid bin.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the single-cell dataset.
    use_rep : str, optional (default: 'X_pca')
        Key in `adata.obsm` for the high-dimensional space used for grid definition and calculation.
    vel_rep : str, optional (default: 'velocity_pca')
        Key in `adata.obsm` containing the RNA velocity vectors.
    n_pca : int, optional (default: 30)
        Number of principal components to use for the high-dimensional vector calculation.
    method : str, optional (default: 'lmoment')
        Method used for calculating the L-Trail directional vector ('lmoment', 'pearson', 'skew').
    grid_size : int, optional (default: 15)
        Number of bins (grid cells) along each of the two dimensions.
    min_cells : int, optional (default: 20)
        Minimum number of cells required in a grid bin to perform the calculation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Grid ID, cell count, center coordinates of the grid, 
        and the cosine similarity score, sorted in descending order of similarity.
    """
    if use_rep not in adata.obsm.keys():
        raise ValueError(f"'{use_rep}' not found in adata.obsm.")
    if vel_rep not in adata.obsm.keys():
        raise ValueError(f"'{vel_rep}' not found in adata.obsm. Did you run scVelo/Velocity first?")

    # 1. Define the spatial grid using the top 2 components
    coords = adata.obsm[use_rep][:, :2]

    # Create evenly spaced bins
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_bins = np.linspace(x_min, x_max, grid_size + 1)
    y_bins = np.linspace(y_min, y_max, grid_size + 1)

    # Determine which grid bin each cell belongs to
    x_indices = np.digitize(coords[:, 0], x_bins)
    y_indices = np.digitize(coords[:, 1], y_bins)
    grid_ids = np.array([f"{x}_{y}" for x, y in zip(x_indices, y_indices)])

    # 2. Retrieve high-dimensional vectors
    X_high = adata.obsm[use_rep][:, :n_pca]
    V_high = adata.obsm[vel_rep][:, :n_pca]
    unique_grids = np.unique(grid_ids)

    results = []

    # Calculate similarity for each grid bin
    for grid_id in unique_grids:
        mask = (grid_ids == grid_id)
        subset_high = X_high[mask]
        subset_vel = V_high[mask]

        # Skip grid bins with insufficient cells
        if len(subset_high) < min_cells:
            continue

        try:
            # Calculate L-Trail vector
            _, vec_ltrail = _calc_high_dim_vector(subset_high, method=method)
            
            # Calculate mean RNA velocity vector
            vec_vel = np.nanmean(subset_vel, axis=0)

            # Calculate magnitudes to filter out zero/noise vectors
            mag_ltrail = np.linalg.norm(vec_ltrail)
            mag_vel = np.linalg.norm(vec_vel)

            if mag_ltrail < 1e-6 or mag_vel < 1e-6:
                continue

            # Calculate Cosine Similarity
            sim = 1.0 - cosine(vec_vel, vec_ltrail)

            # Store the center coordinates of the grid bin
            center_x = np.mean(coords[mask, 0])
            center_y = np.mean(coords[mask, 1])

            results.append({
                'Grid_ID': grid_id,
                'Cell_Count': len(subset_high),
                'Center_X': center_x,
                'Center_Y': center_y,
                'Cos_Similarity': sim
            })
            
        except Exception:
            # Silently skip grids where calculation fails (e.g., math errors on extreme data)
            pass

    # Compile results into a DataFrame
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values(by='Cos_Similarity', ascending=False)
        
    return df_results


#-----------------------------------------------
def calc_knn_similarity(
    adata,
    use_rep: str = 'X_pca',
    vel_rep: str = 'velocity_pca',
    n_pcs: int = 30,
    method: str = 'lmoment',
    k: int = 50,
    n_anchors: int = 1000,
    random_state: int = 34
) -> pd.DataFrame:
    """
    Calculate the local cosine similarity between RNA velocity and L-Trail vectors using a k-NN approach.

    This function samples random anchor cells, identifies their k-nearest neighbors in a 
    high-dimensional space, and compares the mean RNA velocity vector of the neighborhood 
    with the local L-Trail directional vector using cosine similarity.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the single-cell dataset.
    use_rep : str, optional (default: 'X_pca')
        Key in `adata.obsm` for the high-dimensional space used for calculation.
    vel_rep : str, optional (default: 'velocity_pca')
        Key in `adata.obsm` containing the RNA velocity vectors.
    n_pcs : int, optional (default: 30)
        Number of principal components to use from the high-dimensional space.
    method : str, optional (default: 'lmoment')
        Method used for calculating the L-Trail directional vector ('lmoment', 'pearson', 'skew').
    k : int, optional (default: 50)
        Number of nearest neighbors to define the local neighborhood. A relatively large `k` 
        is recommended to maintain the stability and accuracy of the L-Trail calculation.
    n_anchors : int, optional (default: 1000)
        Number of random anchor cells to sample for the evaluation. If None, all cells are used.
    random_state : int, optional (default: 34)
        Random seed for sampling the anchor cells to ensure reproducibility.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Anchor Index, 2D Center coordinates, and the 
        calculated Cosine Similarity for each valid anchor neighborhood.
    """
    if use_rep not in adata.obsm.keys():
        raise ValueError(f"'{use_rep}' not found in adata.obsm.")
    if vel_rep not in adata.obsm.keys():
        raise ValueError(f"'{vel_rep}' not found in adata.obsm. Did you run scVelo/Velocity first?")

    # 1. Retrieve high-dimensional coordinates and velocity vectors
    X_high = adata.obsm[use_rep][:, :n_pcs]
    V_high = adata.obsm[vel_rep][:, :n_pcs]

    # 2D coordinates for visualization purposes (usually PC1 and PC2)
    coords_2d = adata.obsm[use_rep][:, :2]

    n_cells = X_high.shape[0]

    # 2. Sample anchor cells
    np.random.seed(random_state)
    if n_anchors is not None and n_anchors < n_cells:
        anchor_indices = np.random.choice(n_cells, n_anchors, replace=False)
    else:
        anchor_indices = np.arange(n_cells)

    # 3. Build the k-NN graph in the high-dimensional space
    print(f"Building k-NN graph (k={k}) in {use_rep} space...")
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_high)
    _, indices = nbrs.kneighbors(X_high[anchor_indices])

    results = []

    # 4. Calculate similarity for each anchor's neighborhood
    print(f"Calculating similarities for {len(anchor_indices)} anchor points...")
    for i, anchor_idx in enumerate(anchor_indices):
        neighbors_idx = indices[i]  # Indices of the k nearest neighbors including the anchor

        subset_high = X_high[neighbors_idx]
        subset_vel = V_high[neighbors_idx]

        try:
            # Calculate L-Trail vector for the local neighborhood
            _, vec_ltrail = _calc_high_dim_vector(subset_high, method=method)

            # Calculate the mean RNA velocity vector for the local neighborhood
            vec_vel = np.nanmean(subset_vel, axis=0)

            # Calculate magnitudes to filter out zero/noise vectors
            mag_ltrail = np.linalg.norm(vec_ltrail)
            mag_vel = np.linalg.norm(vec_vel)

            if mag_ltrail < 1e-6 or mag_vel < 1e-6:
                continue

            # Calculate Cosine Similarity
            sim = 1.0 - cosine(vec_vel, vec_ltrail)

            results.append({
                'Anchor_Index': anchor_idx,
                'Center_X': coords_2d[anchor_idx, 0],
                'Center_Y': coords_2d[anchor_idx, 1],
                'Cos_Similarity': sim,
            })

        except Exception:
            # Silently skip if calculation fails
            pass

    df_results = pd.DataFrame(results)
    print("Done.")
    
    return df_results
