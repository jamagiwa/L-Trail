#pl

def plot_ltrail(adata,
                groupby,
                basis = 'X_pca',
                use_rep = 'X_pca',
                n_pcs = 30,
                k = 30,
                method = 'lmoment',
                trim = None,
                scale = 5.0,
                p_threshold = 0.05,
                correction = 'bonferroni',
                n_boot = 10000,
                min_cells = None,
                min_vec_length = 0.01,
                legend_loc = 'right margin',
                figsize = (8, 6),
                dot_size = None,
                title = None,
                ax = None,
                alpha = 1,
                show = True,
                save = None,
                sig_table = None,
                random_state = 34,
                frameon = False,
                ):
  """
  Draw cluster-level L-Trail vectors on a 2D embedding via k-NN projection.
 
  trim : int, optional (default : DEFAULT_TRIM)
       TL-moment trimming per side. If 'sig_table' is given, Trim is used.
  min_cells : int, optional (default : DEFAULT_MIN_CELLS)
       If 'sig_table' is given, Min_cells is used.
  p_threshold : float or None, optional (default: 0.05)
       Significance level. Only used when 'sig_table' is None.
       None : draws every finite vector (no test).
  correction : {'bonferroni', 'fdr_bh', None}, optional (default: 'bonferroni')
       Multiple-testing correction across clusters. Only used when 'sig_table' is None.
       None : raw p < p_threshold (no correction; not recommended for figures).
       If 'sig_table' is given, the correction recorded in it is used.
  n_boot : int, optional (default: 10000)
       Only use when 'sig_table' is None.
  sig_table : pd.DataFrame, optional (default : None)
       Output of calc_ltrail_significance_table() + multiple_testing().
       Cluster with Significant == True is draw.
  """
  trim , min_cells = _resolve_params(trim, min_cells)
 
  if correction not in ('bonferroni', 'fdr_bh', None):
      raise ValueError(f"Unknown correction '{correction}'. Expected 'bonferroni', 'fdr_bh' or None.")
 
  # make sig_table 
  if sig_table is None:
       sig_table = calc_ltrail_significance_table(
           adata, groupby,
           use_rep=use_rep,
           n_pcs=n_pcs,
           method=method,
           trim=trim,
           n_boot=n_boot if p_threshold is not None else 1,
           min_cells=min_cells,
           random_state=random_state
       )
       if p_threshold is None:
           sig_table['Significant'] = np.isfinite(sig_table['Magnitude'].astype(float))
           sig_table['Correction'] = 'none (no test)'
       elif correction is None:
           sig_table['Significant'] = sig_table['P_raw'] < p_threshold
           sig_table['Correction'] = 'none'
       else:
           sig_table = multiple_testing(sig_table, method=correction, alpha=p_threshold)
  else:
    # if sig_tabele used -> sig_table as a priority
    for col, given in (('Trim', trim), ('Min_cells', min_cells)):
        if  col in sig_table.columns:
             used = sig_table[col].dropna().unique()
             if len(used) == 1 and int(used[0]) != given and not (col == 'Trim' and method != 'lmoment'):
                 warnings.warn(f"sig_table was computed with {col}={int(used[0])}, "
                               f"but {given} was given. The sig_table value is used.")
 
    if 'Trim' in sig_table.columns and sig_table['Trim'].notna().any():
        trim = int(sig_table['Trim'].dropna().iloc[0])
    if 'Significant' not in sig_table.columns:
        raise ValueError("sig_table has no 'Significant' column. Run multiple_testing() first")
 
    # sig_table のベクトルの次元が n_pcs と合っているか
    vec_dims = {len(v) for v in sig_table['Vector'] if v is not None}
    if vec_dims and vec_dims != {n_pcs}:
        raise ValueError(f"sig_table vectors have {sorted(vec_dims)} dims, but n_pcs={n_pcs}. "
                         f"Use the same n_pcs as in calc_ltrail_significance_table().")
 
  if ax is None:
      fig, ax = plt.subplots(figsize=figsize)
  else:
      fig = ax.get_figure()
 
  _check_obsm(adata, use_rep, basis)
 
  X_high = adata.obsm[use_rep][:, :n_pcs]
  X_vis = adata.obsm[basis][:, :2]
  nbrs = NearestNeighbors(n_neighbors=k).fit(X_high)
 
  #title
  if title is None:
    used_corr = sig_table['Correction'].iloc[0] if 'Correction' in sig_table.columns and len(sig_table) else 'unknown'
    alpha_txt = sig_table['Alpha'].iloc[0] if 'Alpha' in sig_table.columns and len(sig_table) else p_threshold
    if used_corr == 'none (no test)':
        sig_txt = ", no test"
    elif used_corr == 'none':
        sig_txt = f", raw p < {alpha_txt} (uncorrected)"
    else:
        sig_txt = f", {used_corr}-adjusted p < {alpha_txt}"
    trim_txt = f", trim = {trim}" if method == 'lmoment' else ""
    title = f"L-Trail Estimation ({method}{trim_txt}, scale = {scale})\nSpace: {use_rep} -> {basis}{sig_txt}"
 
  scanpy_basis = basis.replace('X_', '', 1) if basis.startswith('X_') else basis
 
  sc.pl.embedding(adata,
                  basis=scanpy_basis,
                  color = groupby,
                  legend_loc = legend_loc,
                  size = dot_size,
                  title = title,
                  ax = ax,
                  show = False,
                  frameon = frameon,
                  alpha = alpha
                  )
 
  #message
  print("--- Processing Information ---")
  print(f"method : {method}")
  if method == 'lmoment':
      print(f"Trim : {trim} per side (0 = L-moment)")
  print(f"Min cells : {min_cells}")
  print(f"Input Space : {use_rep}, (Top {n_pcs} PCs)")
  print(f"Output Space : {basis}")
 
  quiver_X, quiver_Y, quiver_U, quiver_V = [], [], [], []
  for _, row in sig_table[sig_table['Significant'].astype(bool)].iterrows():
      mean_high, vec_high = row['Mean_pos'], row['Vector']
      if mean_high is None or vec_high is None or not np.all(np.isfinite(vec_high)):
          continue
      #calc ltrail vector
      #start is mean in each cluster
      _, idxs_s = nbrs.kneighbors([mean_high])
      start_2d = X_vis[idxs_s[0]].mean(axis=0)
     #end is Vector end
      _, idxs_e = nbrs.kneighbors([mean_high + vec_high * scale])
      end_2d = X_vis[idxs_e[0]].mean(axis=0)
 
      # 2d vector
      dx, dy = end_2d - start_2d
      # hypot() is calc hypotenuse euclid distance
      if np.hypot(dx, dy) < min_vec_length:
          print(f"Hidden {row['Cluster']}: projection length < min_vec_length")
          continue
 
      quiver_X.append(start_2d[0]); quiver_Y.append(start_2d[1])
      quiver_U.append(dx); quiver_V.append(dy)
      ax.scatter(start_2d[0], start_2d[1], c='#222222', edgecolors='white', linewidths=0.5, s=15, zorder=11)
 
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
                edgecolors='white',
                linewidth=0.5,
                headwidth=5,
                headlength=6,
                headaxislength=5,
                zorder=12
                )
  else:
      print('Warning : No vectors to draw. Check thresholds or data')
 
  if save is not None:
      fig.savefig(save, bbox_inches='tight')
      print(f"Saved figure to : {save}")
 
  if show:
      plt.show()
      return None
  return fig

def plot_knn_similarity(adata, 
                        df_results: pd.DataFrame, 
                        basis: str = 'X_pca', 
                        title: str = None, 
                        figsize: tuple = (8, 6), 
                        show : bool = True, 
                        ):
  
  """
  Visualize the k-NN  similarity on a low-dimensional embedding space.
  
  Parameter
  ----------
  adata : Anndata
     Annotated data matrix containing the single-cell dataset.
  df_results : pd.DataFrame
     DataFrame containing the local similarity results, typically generated by `calc_knn_similarity`.
  basis : str, optional (default: 'X_pca')
     Key in `adata.obsm` used for plotting. Anchor positions are taken from this
     embedding via 'Anchor_Index', so they always match the background cells.
  title : str, optional (default: None)
     Title of the plot. If None, a default descriptive title is used.
  figsize : tuple, optional (default: (8, 6))
     Width and height of the figure in inches.
  show : bool, optional (default: True)
     Whether to display the figure immediately. If False, return the axes object.

  Return
  ----------
  ax : matplotlib.axes.Axes or None
       Returns the matplotlib Axes object if `show = False`, else return None.
  """

  #Check for empty data
  if df_results.empty:
      print("Warning: No data to plot in plot_knn_similarity")
      return

  fig, ax = plt.subplots(figsize = figsize)

  #Plot background cells
  if basis not in adata.obsm:
      raise ValueError(f"'{basis}' not found in adata.obsm.")
  coords_bg = adata.obsm[basis][:, :2]
  ax.scatter(coords_bg[:, 0], 
             coords_bg[:, 1], 
             c = 'lightgray', 
             alpha = 0.2, 
             s = 10, 
             zorder = 1, 
             )


  #Plot evaluated anchor cells colored by cosine similarity
  # Center_X/Y は calc_knn_similarity の use_rep (通常 PCA) の座標なので、
  # basis が UMAP などのときにずれる。Anchor_Index から basis の座標を取り直す。
  anchor_xy = coords_bg[df_results['Anchor_Index'].to_numpy(dtype=int)]
  scat = ax.scatter(anchor_xy[:, 0], 
                    anchor_xy[:, 1], 
                    c = df_results['Cos_Similarity'], 
                    cmap = 'coolwarm', 
                    vmin = -1.0, 
                    vmax = 1.0, 
                    s = 30, 
                    edgecolor = 'black', 
                    linewidth = 0.5, 
                    zorder = 2
                    )
  
  #Add colorbar
  plt.colorbar(scat, 
               ax = ax, 
               label = 'Cosine similarity (Velocity vs L-Trail)')
  
  #Title
  if title:
      ax.set_title(title)
  else:
      ax.set_title("k-NN based Vector Similarity")
  
  #Set axis labels
  scanpy_basis = basis.replace('X_', '') if basis.startswith('X_') else basis
  ax.set_xlabel(f"{scanpy_basis}1")
  ax.set_ylabel(f"{scanpy_basis}2")

  if show:
      plt.show()
      return None
  
  return ax

def boxplot_similarity(adata, 
                      df_results: pd.DataFrame, 
                      groupby: str = 'clusters', 
                      title: str = None, 
                      figsize: tuple = (10, 6), 
                      ylim: tuple = None, 
                      show: bool = True
                      ):
    """
 
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
    summary_stats = df_plot.groupby('Cluster', observed=True)['Cos_Similarity'].agg(['count', 'median']).sort_values(by='median', ascending=False)
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
