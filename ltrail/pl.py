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

 #要検討
def random_direction_null(n_dims: int, n_draw: int = 100_000, q=(0.025, 0.975), seed: int = 0):
    """
    n_dims 次元のランダムな方向どうしの cos の分位点 (既定は 95% 範囲)。
    観測された cos がこの範囲に入るなら、ランダムな矢印と区別できない。
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_draw, n_dims)); b = rng.normal(size=(n_draw, n_dims))
    c = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return tuple(float(x) for x in np.quantile(c, q))
 

def boxplot_similarity(adata,
                       df_results: pd.DataFrame = None,
                       groupby: str = 'clusters',
                       title: str = None,
                       figsize: tuple = (10, 6),
                       ylim: tuple = (-1.05, 1.05),
                       show: bool = True,
                       df_cluster: pd.DataFrame = None,
                       sig_table: pd.DataFrame = None,
                       order='auto',
                       show_null: bool = True,
                       n_pcs: int = None,
                       ):
    """
    Cosine similarity between RNA velocity and L-Trail vectors, per cluster:
    k-NN neighborhoods (violin + box) and/or the cluster-level value (one marker per cluster).
 
    Parameters
    ----------
    adata : AnnData
    df_results : pd.DataFrame or None
        Output of calc_knn_similarity() (needs 'Anchor_Index'). Drawn as violin + box.
    groupby : str, optional (default: 'clusters')
    title : str, optional
    figsize : tuple, optional (default: (10, 6))
    ylim : tuple, optional (default: (-1.05, 1.05))
    show : bool, optional (default: True)
        If False, returns the Axes object.
    df_cluster : pd.DataFrame or None
        Output of calc_velocity_ltrail_similarity(). Drawn as one diamond per cluster.
    sig_table : pd.DataFrame or None
        Output of calc_ltrail_significance_table() + multiple_testing(). If given, the
        cluster-level marker is filled for significant clusters and open otherwise.
    order : 'auto' | 'cluster' | 'knn' | 'category' | list, optional (default: 'auto')
        'auto' sorts by the cluster-level cosine if df_cluster is given, else by the k-NN median.
        'category' keeps the order of adata.obs[groupby].
    show_null : bool, optional (default: True)
        Shade the 95% range of the cosine between two random directions in n_pcs dimensions.
    n_pcs : int or None
        Dimensionality for the null range. Defaults to df_cluster['N_pcs'] (or 30).
 
    Returns
    -------
    ax : matplotlib.axes.Axes or None
    """
    if df_results is None and df_cluster is None:
        raise ValueError("Give df_results (k-NN), df_cluster (cluster level), or both.")
    if groupby not in adata.obs:
        raise ValueError(f"'{groupby}' not found in adata.obs.")
 
    # --- k-NN: アンカー細胞をクラスタに対応づける
    knn = None
    if df_results is not None:
        if 'Anchor_Index' not in df_results.columns:
            raise ValueError("'Anchor_Index' column not found in df_results.")
        knn = df_results[['Anchor_Index', 'Cos_Similarity']].copy()
        knn['Cluster'] = adata.obs[groupby].iloc[knn['Anchor_Index'].to_numpy(dtype=int)].astype(str).to_numpy()
        knn_stats = knn.groupby('Cluster')['Cos_Similarity'].agg(['count', 'median'])
    else:
        knn_stats = pd.DataFrame(columns=['count', 'median'])
 
    # --- クラスタ単位の値
    clus = pd.Series(dtype=float)
    if df_cluster is not None:
        clus = df_cluster.assign(Cluster=df_cluster['Cluster'].astype(str)).set_index('Cluster')['Cos_Similarity']
    sig_set = None
    if sig_table is not None:
        if 'Significant' not in sig_table.columns:
            raise ValueError("sig_table has no 'Significant' column. Run multiple_testing() first.")
        sig_set = set(sig_table.loc[sig_table['Significant'].astype(bool), 'Cluster'].astype(str))
 
    # --- 並び順 (どちらか一方にしかないクラスタも含める)
    g = adata.obs[groupby]
    categories = [str(c) for c in (g.cat.categories if hasattr(g, 'cat') else pd.unique(g))]
    present = [c for c in categories if c in set(knn_stats.index) | set(clus.index)]
    if isinstance(order, (list, tuple)):
        clusters = [str(c) for c in order]
    elif order == 'category':
        clusters = present
    elif order == 'cluster' or (order == 'auto' and df_cluster is not None):
        clusters = sorted(present, key=lambda c: -np.nan_to_num(clus.get(c, np.nan), nan=-9))
    elif order in ('knn', 'auto'):
        clusters = sorted(present, key=lambda c: -np.nan_to_num(knn_stats['median'].get(c, np.nan), nan=-9))
    else:
        raise ValueError(f"Unknown order '{order}'.")
    if not clusters:
        print("Warning: No data to plot.")
        return None
 
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(clusters))
    handles = []
 
    # --- ランダム方向の帰無範囲
    if show_null:
        if n_pcs is None:
            n_pcs = int(df_cluster['N_pcs'].iloc[0]) if df_cluster is not None and 'N_pcs' in df_cluster else 30
        lo, hi = random_direction_null(n_pcs)
        handles.append(ax.axhspan(lo, hi, color='#bdbdbd', alpha=0.35, lw=0, zorder=0,
                                  label=f'Random-direction 95% range ({n_pcs} dims)'))
    ax.axhline(0, color='#555555', linestyle='--', linewidth=1, zorder=1)
 
    # --- k-NN の分布 (violin + box)
    if knn is not None:
        data = [knn.loc[knn['Cluster'] == c, 'Cos_Similarity'].dropna().to_numpy() for c in clusters]
        has = [i for i, d in enumerate(data) if len(d) > 1]
        if has:
            vp = ax.violinplot([data[i] for i in has], positions=x[has], widths=0.8,
                               showextrema=False)
            for b in vp['bodies']:
                b.set_facecolor('#d9d9d9'); b.set_edgecolor('none'); b.set_alpha(0.6); b.set_zorder(2)
            bp = ax.boxplot([data[i] for i in has], positions=x[has], widths=0.25, showfliers=False,
                            patch_artist=True, zorder=3,
                            boxprops=dict(facecolor='#2a78d6', edgecolor='#1c5cab', alpha=0.85),
                            medianprops=dict(color='white', linewidth=1.5),
                            whiskerprops=dict(color='#1c5cab'), capprops=dict(color='#1c5cab'))
            from matplotlib.patches import Patch
            handles.append(Patch(facecolor='#2a78d6', edgecolor='#1c5cab',
                                 label='k-NN neighborhoods (one value per anchor cell)'))
 
    # --- クラスタ単位 (1 クラスタ 1 点)
    if df_cluster is not None:
        y = np.array([clus.get(c, np.nan) for c in clusters], dtype=float)
        filled = np.array([sig_set is None or c in sig_set for c in clusters])
        ax.scatter(x[filled], y[filled], marker='D', s=70, color='#eb6834', edgecolor='white',
                   linewidth=1.0, zorder=5)
        ax.scatter(x[~filled], y[~filled], marker='D', s=70, facecolor='white', edgecolor='#eb6834',
                   linewidth=1.5, zorder=5)
        from matplotlib.lines import Line2D
        lab = 'Cluster level (one value per cluster)'
        handles.append(Line2D([], [], marker='D', ls='none', markersize=8, color='#eb6834',
                              markeredgecolor='white', label=lab + (', significant' if sig_set is not None else '')))
        if sig_set is not None:
            handles.append(Line2D([], [], marker='D', ls='none', markersize=8, markerfacecolor='white',
                                  markeredgecolor='#eb6834', label='Cluster level, not significant'))
 
    # --- 軸ラベル (k-NN の n と中央値、クラスタ単位の値)
    labels = []
    for c in clusters:
        parts = [c]
        if knn is not None:
            if c in knn_stats.index:
                parts.append(f"k-NN med {knn_stats.loc[c, 'median']:.2f} (n={int(knn_stats.loc[c, 'count'])})")
            else:
                parts.append("k-NN: no anchors")
        if df_cluster is not None:
            v = clus.get(c, np.nan)
            parts.append(f"cluster {v:.2f}" if np.isfinite(v) else "cluster: n/a")
        labels.append("\n".join(parts))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlim(-0.6, len(clusters) - 0.4)
    ax.set_ylabel('Cosine similarity (velocity vs L-Trail)', fontsize=11)
    ax.set_xlabel('')
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, axis='y', color='#e5e5e5', linewidth=0.6); ax.set_axisbelow(True)
    sns.despine(ax=ax)
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=9)
    if title:
        ax.set_title(title, fontsize=13, pad=48 if handles else 12)
    fig.tight_layout()
 
    if show:
        plt.show()
        return None
    return ax
