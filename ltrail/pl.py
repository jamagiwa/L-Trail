import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import seaborn as sns
from ltrail import tl

#-----------------------------------------------
def plot_ltrail(adata,
               groupby,
               basis='X_pca',
               use_rep='X_pca',
               n_pcs=30,
               k=30,
               method='lmoment',
               scale=5.0,
               p_threshold=0.05,
               n_boot=1000,
               min_cells=30,
               legend_loc='right margin',
               figsize=(8, 6),
               dot_size=None,
               title=None,
               ax=None,
               alpha=1,
               show=True,
               save=None):
    """
    Project the L-Trail calculated in a high-dimensional space onto an arbitrary low-dimensional embedding.
    
    This function calculates the trajectory vector for each cluster in the high-dimensional PCA space 
    and maps it to a low-dimensional visualization space (e.g., UMAP or PCA) using a k-Nearest 
    Neighbors (k-NN) approach.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the single-cell dataset.
    groupby : str
        Key in `adata.obs` defining the clusters for vector calculation.
    basis : str, optional (default: 'X_pca')
        Key in `adata.obsm` for the low-dimensional embedding used for visualization.
    use_rep : str, optional (default: 'X_pca')
        Key in `adata.obsm` for the high-dimensional space used for vector calculation.
    n_pcs : int, optional (default: 30)
        Number of principal components to use from the high-dimensional space.
    k : int, optional (default: 30)
        Number of nearest neighbors used for the projection mapping.
    method : str, optional (default: 'lmoment')
        Method used for calculating the directional vector ('lmoment', 'pearson', 'skew').
    scale : float, optional (default: 5.0)
        Scaling factor for the length of the drawn vectors.
    p_threshold : float or None, optional (default: 0.05)
        P-value threshold for the permutation test. Vectors with a p-value above this 
        threshold will not be drawn. Set to None to bypass the significance test.
    n_boot : int, optional (default: 100)
        Number of bootstrap iterations for the permutation test.
    min_cells : int, optional (default: 20)
        Minimum number of cells required in a cluster to calculate its vector.
    legend_loc : str, optional (default: 'right margin')
        Location of the legend in the scatter plot.
    figsize : tuple, optional (default: (8, 6))
        Width and height of the figure in inches.
    dot_size : int, optional (default: None)
        Point size for the background scatter plot.
    title : str, optional (default: None)
        Title of the plot.
    ax : matplotlib.axes.Axes, optional (default: None)
        A matplotlib axes object to draw the plot on. If None, a new figure is created.
    alpha : float, optional (default: 0.3)
        Alpha blending value for the background scatter plot points.
    show : bool, optional (default: True)
        Whether to display the figure immediately.
    save : str, optional (default: None)
        Filename to save the figure (e.g., 'figure.pdf').

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Returns the matplotlib Figure object if `show=False`, else returns None.
    """

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show_plot = True
    else:
        fig = ax.get_figure()

    # Check data consistency
    if use_rep not in adata.obsm.keys():
        raise ValueError(f"Calculation space '{use_rep}' not found in adata.obsm.")
    if basis not in adata.obsm.keys():
        raise ValueError(f"Visualization space '{basis}' not found in adata.obsm.")

    # 1. High-dimensional computation space
    X_high = adata.obsm[use_rep][:, :n_pcs]

    # 2. Low-dimensional visualization space
    X_vis = adata.obsm[basis][:, :2]

    # Build k-NN graph for projection
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_high)

    groups = adata.obs[groupby]
    unique_groups = groups.unique()

    if title is None:
        sig_txt = f", p<{p_threshold}" if p_threshold else ""
        title = f"L-Trail Estimation ({method}, scale={scale})\nSpace: {use_rep} -> {basis}{sig_txt}"

    scanpy_basis = basis.replace('X_', '') if basis.startswith('X_') else basis

    # Background scatter plot
    sc.pl.embedding(adata,
                    basis=scanpy_basis,
                    color=groupby,
                    legend_loc=legend_loc,
                    size=dot_size,
                    title=title,
                    ax=ax,
                    show=False,
                    frameon=True,
                    alpha=alpha)

    print(f"--- Processing Information ---")
    print(f"Method       : {method}")
    print(f"Input Space  : {use_rep} (Top {n_pcs} PCs)")
    print(f"Output Space : {basis} (Projected via k-NN)")

    # Calculate vectors for each cluster
    quiver_X, quiver_Y = [], [] # Start points
    quiver_U, quiver_V = [], [] # Vector components (dx, dy)

    for group in unique_groups:

        mask = (groups == group).values
        subset_high = X_high[mask]

        if len(subset_high) < min_cells:
            continue

        try:
            # Step A: Calculate vector in high-dimensional space
            mean_high, vec_high =tl._calc_high_dim_vector(subset_high, method=method)
            magnitude = np.sqrt(np.sum(vec_high**2))

            # Step B: Significance test
            if p_threshold is not None:
                if magnitude < 1e-6:
                    continue

                p_val = tl._test_significance_high_dim(subset_high,
                                                    observed_magnitude=magnitude,
                                                    method=method,
                                                    n_boot=n_boot)
                if p_val >= p_threshold:
                    continue # Skip if not statistically significant

            # Step C: Projection (k-NN Mapping)
            # Find the projected start point
            dists_s, idxs_s = nbrs.kneighbors([mean_high])
            start_2d = np.mean(X_vis[idxs_s[0]], axis=0)

            # Find the projected end point
            future_high = mean_high + (vec_high * scale)
            dists_e, idxs_e = nbrs.kneighbors([future_high])
            end_2d = np.mean(X_vis[idxs_e[0]], axis=0)

            # Calculate 2D vector components
            dx = end_2d[0] - start_2d[0]
            dy = end_2d[1] - start_2d[1]

            # Remove negligible noisy vectors
            if np.sqrt(dx**2 + dy**2) < 0.01:
                continue

            # Append to drawing lists
            quiver_X.append(start_2d[0])
            quiver_Y.append(start_2d[1])
            quiver_U.append(dx)
            quiver_V.append(dy)

            # Highlight the start points of the vectors
            ax.scatter(start_2d[0],
                       start_2d[1],
                       c='#222222',
                       edgecolor='white',
                       linewidth=0.5,
                       s=15,
                       zorder=11
                       )

        except Exception as e:
            print(f"Skipped {group}: {e}")

    # Draw the vectors using matplotlib quiver
    if quiver_X:
        print(f"Drawing {len(quiver_X)} vectors...")
        ax.quiver(quiver_X,
                  quiver_Y,
                  quiver_U,
                  quiver_V,
                  angles='xy',
                  scale_units='xy',
                  scale=1,
                  color='#222222',
                  edgecolor='white',
                  linewidth=0.5,
                  headwidth=5,
                  headlength=6,
                  headaxislength=5,
                  zorder=12)
    else:
        print("Warning: No vectors to draw! Check thresholds or data.")

    # Handle saving and display
    if save is not None:
        fig.savefig(save, bbox_inches='tight')
        print(f"Saved figure to: {save}")

    if show:
        plt.show()
        return None

    return fig

#-----------------------------------------------
def plot_grid_similarity_map(
    adata,
    df_grid_sim: pd.DataFrame,
    basis: str = 'X_pca',
    title: str = None,
    figsize: tuple = (8, 6),
    show: bool = True
):
    """
    Visualize the grid-based cosine similarity map on a low-dimensional embedding space.

    This function plots the background cells and overlays the center of each spatial grid.
    The color of the grid points represents the cosine similarity between RNA velocity 
    and L-Trail vectors, and the size of the points is scaled by the number of cells in that grid.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the single-cell dataset.
    df_grid_sim : pd.DataFrame
        DataFrame containing grid similarity results, typically generated by `calc_grid_similarity`.
    basis : str, optional (default: 'X_pca')
        Key in `adata.obsm` used for plotting the background cells.
    title : str, optional (default: None)
        Title of the plot. If None, a default descriptive title is used.
    figsize : tuple, optional (default: (8, 6))
        Width and height of the figure in inches.
    show : bool, optional (default: True)
        Whether to display the figure immediately. If False, returns the Axes object.

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        Returns the matplotlib Axes object if `show=False`, else returns None.
    """
    if df_grid_sim.empty:
        print("Error: No data to plot.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # 1. Plot background cells
    if basis in adata.obsm:
        coords = adata.obsm[basis]
        ax.scatter(coords[:, 0],
                   coords[:, 1],
                   c='lightgray',
                   s=10,
                   alpha=0.3,
                   label='Background Cells',
                   zorder=1)
    else:
        print(f"Warning: '{basis}' not found in adata.obsm. Background cells skipped.")

    # 2. Plot grid centers
    # Scale point sizes based on the cell count within each grid
    sizes = df_grid_sim['Cell_Count'] * 3

    scatter = ax.scatter(df_grid_sim['Center_X'],
                         df_grid_sim['Center_Y'],
                         c=df_grid_sim['Cos_Similarity'],
                         s=sizes,
                         cmap='coolwarm',
                         vmin=-1.0,
                         vmax=1.0,
                         edgecolors='black',
                         linewidths=0.5,
                         zorder=5)

    # 3. Aesthetics and Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cosine Similarity (Velocity vs L-Trail)')

    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Grid-based Cosine Similarity Map\nRed: Aligned (>0), Blue: Opposing (<0)")

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(False)

    if show:
        plt.show()
        return None
    
    return ax

#-----------------------------------------------
def plot_knn_similarity(
    adata,
    df_results: pd.DataFrame,
    basis: str = 'X_pca',
    title: str = None,
    figsize: tuple = (8, 6),
    show: bool = True
):
    """
    Visualize the k-NN based local cosine similarity on a low-dimensional embedding space.

    This function plots the background cells from the single-cell dataset and overlays 
    the anchor cells evaluated in `calc_knn_similarity`. The anchor cells are colored 
    according to their cosine similarity between RNA velocity and L-Trail vectors.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the single-cell dataset.
    df_results : pd.DataFrame
        DataFrame containing the local similarity results, typically generated by `calc_knn_similarity`.
    basis : str, optional (default: 'X_pca')
        Key in `adata.obsm` used for plotting the background cells.
    title : str, optional (default: None)
        Title of the plot. If None, a default descriptive title is used.
    figsize : tuple, optional (default: (8, 6))
        Width and height of the figure in inches.
    show : bool, optional (default: True)
        Whether to display the figure immediately. If False, returns the Axes object.

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        Returns the matplotlib Axes object if `show=False`, else returns None.
    """
    # Check for empty data
    if df_results.empty:
        print("Warning: No data to plot in plot_knn_similarity.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # 1. Plot background cells
    if basis in adata.obsm:
        coords_bg = adata.obsm[basis][:, :2]
        ax.scatter(coords_bg[:, 0],
                   coords_bg[:, 1],
                   c='lightgray',
                   alpha=0.2,
                   s=10,
                   zorder=1)
    else:
        print(f"Warning: '{basis}' not found in adata.obsm. Background cells skipped.")

    # 2. Plot evaluated anchor cells colored by cosine similarity
    scat = ax.scatter(
        df_results['Center_X'],
        df_results['Center_Y'],
        c=df_results['Cos_Similarity'],
        cmap='coolwarm',  # Blue (opposing) - White (orthogonal) - Red (aligned)
        vmin=-1.0,
        vmax=1.0,
        s=30,
        edgecolor='black',
        linewidth=0.5, 
        zorder=2
    )

    # Add colorbar
    plt.colorbar(scat, ax=ax, label='Cosine Similarity (Velocity vs L-Trail)')

    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title("k-NN based Vector Similarity (Manifold-aware)")

    # Set axis labels
    scanpy_basis = basis.replace('X_', '') if basis.startswith('X_') else basis
    ax.set_xlabel(f"{scanpy_basis}1")
    ax.set_ylabel(f"{scanpy_basis}2")

    if show:
        plt.show()
        return None
        
    return ax

#-----------------------------------------------
def boxplot_similarity(
    adata,
    df_results: pd.DataFrame,
    groupby: str = 'clusters',
    title: str = None,
    figsize: tuple = (10, 6),
    ylim: tuple = None,
    show: bool = True
):
    """
    Draw a publication-ready combined Violin and Box plot of cosine similarities.

    This function visualizes the distribution of cosine similarities between RNA velocity 
    and L-Trail vectors across different cell clusters. It automatically calculates and 
    displays the sample size (n) and median value for each cluster on the x-axis labels.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the single-cell dataset.
    df_results : pd.DataFrame
        DataFrame containing the local similarity results, typically generated by `calc_knn_similarity`.
        Must contain the 'Anchor_Index' column.
    groupby : str, optional (default: 'clusters')
        Key in `adata.obs` defining the clusters to group the similarities by.
    title : str, optional (default: None)
        Title of the plot.
    figsize : tuple, optional (default: (10, 6))
        Width and height of the figure in inches.
    ylim : tuple, optional (default: None)
        Y-axis limits. For cosine similarity, (-1.0, 1.0) or (-1.1, 1.1) is recommended.
    show : bool, optional (default: True)
        Whether to display the figure immediately. If False, returns the Axes object.

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        Returns the matplotlib Axes object if `show=False`, else returns None.
    """
    # 1. Set plot style
    sns.set_style("ticks")

    df_plot = df_results.copy()

    # Data validation
    if 'Anchor_Index' not in df_plot.columns:
        print("Error: 'Anchor_Index' column not found in df_results.")
        return
    if groupby not in adata.obs:
        print(f"Error: '{groupby}' not found in adata.obs.")
        return

    # Map cluster information from adata to the results DataFrame
    try:
        indices = df_plot['Anchor_Index'].values.astype(int)
        cluster_labels = adata.obs[groupby].iloc[indices].values
        df_plot['Cluster'] = cluster_labels
    except Exception as e:
        print(f"Error mapping clusters: {e}")
        return

    # Calculate summary statistics to order the plot by median (descending)
    summary_stats = df_plot.groupby('Cluster')['Cos_Similarity'].agg(['count', 'median']).sort_values(by='median', ascending=False)
    median_order = summary_stats.index

    if len(median_order) == 0:
        print("Warning: No data to plot.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Violin plot (background distribution)
    sns.violinplot(
        data=df_plot,
        x='Cluster',
        y='Cos_Similarity',
        order=median_order,
        inner=None,
        color='lightgray',
        linewidth=0,
        alpha=0.4,
        ax=ax
    )

    # Box plot (foreground statistics)
    sns.boxplot(
        data=df_plot,
        x='Cluster',
        y='Cos_Similarity',
        order=median_order,
        width=0.25,
        color='#4C72B0',
        showfliers=False,
        linewidth=1.2,
        ax=ax
    )

    # Draw reference line at 0 (orthogonal vectors)
    ax.axhline(0, color='#333333', linestyle='--', linewidth=1, alpha=0.8)

    # Set title and axis labels
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    ax.set_ylabel('Cosine similarity', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cell groups', fontsize=12, fontweight='bold')

    # Append n-count and median to X-axis labels
    new_labels = []
    for cluster in median_order:
        count = int(summary_stats.loc[cluster, 'count'])
        median_val = summary_stats.loc[cluster, 'median']
        new_labels.append(f"{cluster}\n(n={count})\nMed: {median_val:.2f}")

    ax.set_xticks(range(len(median_order)))
    ax.set_xticklabels(new_labels, rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Remove top and right spines
    sns.despine(ax=ax)

    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()

    if show:
        plt.show()
        return None
        
    return ax
