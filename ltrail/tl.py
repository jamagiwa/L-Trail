#tl
def _cos_sim(a: np.ndarray,
             b: np.ndarray,
             eps: float = 1e-6) -> float:
             """Calclate cosine similarity. NaN if either vector is non-finite or ~0"""
             #linalg is scipy　matrix
             #calc l2 norm
             na, nb = np.linalg.norm(a), np.linalg.norm(b)
             if not (np.isfinite(na) and np.isfinite(nb)) or na < eps or nb < eps:
               return np.nan
             return float(a @ b / (na * nb))


def _check_obsm(adata, *keys):
  for key in keys:
    if key not in adata.obsm.keys():
      raise ValueError(f"'{key}' not found in adata.obsm.")


#pca空間が一致するか確認するための関数
#-------------------------------------------
#pca毎にhashを作る。同じpca空間かの判定を行う。
def _basis_fingerprint(adata, use_rep):
    if use_rep == 'X_pca' and 'PCs' in getattr(adata, 'varm', {}):
        src = np.asarray(adata.varm['PCs'], dtype=np.float64)
    else:
        src = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    return hashlib.sha1(np.ascontiguousarray(src).tobytes()).hexdigest()[:16]


def register_velocity_basis(
  adata,
  use_rep: str = 'X_pca',
  vel_rep: str = 'velocity_pca'
):
  """
  veloでuse_repに射影した直後に、射影先のpcaを記録。
  calc_*_similarity()は記録した現在のuse_repを照合、PCAが再計算されてたらエラー

  for instance
  sc.tl.pca(adata)
  scv.tl.velocity_embedding(adata, basis='pca')
  ltrail.register_velocity_basis(adata) # <- record
  """
  _check_obsm(adata, use_rep, vel_rep)
  #setdefault is add data in dic
  adata.uns.setdefault('ltrail_velocity_basis', {})[vel_rep] = {
    'use_rep': use_rep,
    'fingerprint': _basis_fingerprint(adata, use_rep),
    'n_cols': int(adata.obsm[vel_rep].shape[1]),
  }

#############SIGNIGICANT###################
def _get_paired_reps(
  adata,
  use_rep: str,
  vel_rep: str,
  n_pcs: int
  ):
  """
  L-Trailとvelocityを同じPCA空間の同じn_pcs次元で取り出す。
  """
  _check_obsm(adata, use_rep, vel_rep)
#X is ltrail, Y is Velocity
  X_full, V_full = adata.obsm[use_rep], adata.obsm[vel_rep]

#check (X_pcs and velocity_pcs > n_pcs)
  for key , arr in ((use_rep, X_full), (vel_rep, V_full)):
    if arr.shape[1] < n_pcs:
      raise ValueError(f"'{key}' has only {arr.shape[1]} columns (< n_pcs = {n_pcs})."
                      f"Use the same n_pcs for both, or projection velocity onto all PCs"
                      )
#getattr is 属性の取り出し
#adataのunsを取り出し、デフォルトは{}
  record = getattr(adata, 'uns', {}).get('ltrail_velocity_basis', {}).get(vel_rep)
  if record is None:
    warnings.warn(f"Cannot verify that '{vel_rep}'")

  else:
      if record['use_rep'] != use_rep:
        raise ValueError(f"'{vel_rep}' was registerd for '{record['use_rep']}' not '{use_rep}'.")
      if record['fingerprint'] != _basis_fingerprint(adata, use_rep):
        raise ValueError(f"'{use_rep}' has change since '{vel_rep} was compute'")


  X_high, V_high = X_full[:, :n_pcs], V_full[:, :n_pcs]
  bad = ~np.all(np.isfinite(V_high), axis=1)
  if bad.mean() > 0.1:
    warnings.warn(f"{bad.mean():.0%} of cells have non-finite values in '{vel_rep}'.", stacklevel=3)

  return X_high, V_high

#-------------------------------------------

def calc_velocity_ltrail_similarity(
  adata,
  groupby: str,
  use_rep: str = 'X_pca',
  vel_rep: str = 'velocity_pca',
  method: str = 'lmoment',
  trim: int = None,
  n_pcs: int = 30,
  min_cells: int = None,
) -> pd.DataFrame:
  """
  Cosine similarity between the mean RNA velocity vector vs L-trail vector of each cluster.

  Parameters
  ----------
  adata : AnnData
  groupby : str
      Key in `adata.obs` defining the cluster.
  use_rep : str optional (default : 'X_pca')
  vel_rep : str, optional (default : 'velocity_pca')
      Velocity vector in the same space as `use_rep` (needs >= n_pcs columns).
  method : str, optional (default : 'lmoment')
  trim : int, optional (default : DEFAULT_TRIM)
  n_pcs : int, optional (default : 30)
  min_cells : int, optional (default : DEFAULT_MIN_CELLS)
      Cluster with fewer cells are kept in the table with Note='n < min_cells'.

 Return
 pd.DataFrame : Cluster, N_cells, Trim, Min_cells, Cos_Similarity, Note
  """
  trim, min_cells = _resolve_params(trim, min_cells)
  X_high, V_high = _get_paired_reps(adata, use_rep, vel_rep, n_pcs)
  groups = adata.obs[groupby]

  results = []

  for group in groups.unique():
    mask = (groups == group).values
    n_cells = int(mask.sum())
    row = {'Cluster': group,
           'N_cells': n_cells,
           'Trim': trim if method == 'lmoment' else np.nan,
           'Min_cells': min_cells,
           'N_pcs': n_pcs}
    # min_cells cos sim = nan
    if n_cells < min_cells:
      #append is add list
      results.append({**row, 'Cos_Similarity': np.nan, 'Note': 'n < min_cells'})
      continue

    _, vec_ltrail = _calc_high_dim_vector(X_high[mask],
                                          method=method,
                                          trim=trim,
                                          min_cells=min_cells
                                         )
#velocity mean vector / mask(for instance; /cluster)
    vec_vel = np.nanmean(V_high[mask], axis=0)
#calc cos similarity
    sim = _cos_sim(vec_vel, vec_ltrail)

    results.append({**row, 'Cos_Similarity': sim,
                'Note': 'Success' if np.isfinite(sim) else 'zero vector'})

  df = pd.DataFrame(results)
  if not df.empty:
    df = df.sort_values('Cos_Similarity', ascending=False, na_position='last')

  return df.reset_index(drop=True)


##grid similarityは排除

def calc_knn_similarity(adata,
                        use_rep: str = 'X_pca',
                        vel_rep: str = 'velocity_pca',
                        n_pcs: int = 30,
                        method: str = 'lmoment',
                        trim: int = None,
                        k: int = 50,
                        n_anchors: int = 1000,
                        random_state: int = 34,
                        ) -> pd.DataFrame:
                        """
                        Local cosine similarity between RNA velocity and L-Trail in the k-NN neighborhoods of random sampled anchor cells.
                        k: int, optional (default = 50)
                        Neighborhood size. Must be >=DEFAULT_MIN_CELLS.
                        """

                        trim, min_cells = _resolve_params(trim, None)

                        if k < min_cells:
                          raise ValueError(f"'k={k}' is smaller than min_cells={min_cells}'")

                        X_high, V_high = _get_paired_reps(adata, use_rep, vel_rep, n_pcs)
                        coords_2d = adata.obsm[use_rep][:, :2]
                        n_cells = X_high.shape[0]

                        #Global random seed is constant
                        #generate random　numbers
                        rng = np.random.default_rng(random_state)

                        if n_anchors is not None and n_anchors < n_cells:
                          anchor_indices = rng.choice(n_cells, n_anchors, replace=False)
                        else:
                          anchor_indices = np.arange(n_cells)

                        print(f"Calclating k-NN graph (k={k}) in {use_rep} space...")

                        nbrs = NearestNeighbors(n_neighbors=k).fit(X_high)
                        _, indices = nbrs.kneighbors(X_high[anchor_indices])

                        print(f"Calclating similarities for {len(anchor_indices)} anchor points...")

                        results = []

                        for anchor_idx, nb_idx in zip(anchor_indices, indices):
                          _, vec_ltrail = _calc_high_dim_vector(X_high[nb_idx],
                                                                method=method,
                                                                trim=trim,
                                                                min_cells=min_cells)
                          #Velocity　must be calclate mean
                          sim = _cos_sim(np.nanmean(V_high[nb_idx], axis=0), vec_ltrail)
                          if not np.isfinite(sim):
                            continue
                          results.append({'Anchor_Index': anchor_idx,
                                          'Center_X': coords_2d[anchor_idx, 0],
                                          'Center_Y': coords_2d[anchor_idx, 1],
                                          'Cos_Similarity': sim})
                        print("Done.")
                        return pd.DataFrame(results)

###要検討
#どの軸でdiagramを作成するか考える必要がある
def calc_cluster_l_moment_diagram(
    adata,
    groupby: str,
    use_rep: str = 'X_pca',
    n_pcs: int = 30,
    method: str = 'lmoment',
    trim: int = None,
    min_cells: int = None,
    max_order: int = 4,
) -> pd.DataFrame:
    """
    各クラスタの L-Trail 方向ベクトルに細胞を射影した1次元分布について、
    L-skewness (tau3)・L-kurtosis (tau4)、必要ならさらに高次の tau を計算する。

    Returns
    -------
    pd.DataFrame : Cluster, N_cells, Trim, Min_cells, l2, tau3, ..., Note
    """
    trim, min_cells = _resolve_params(trim, min_cells, max_order=max_order)
    X_high = adata.obsm[use_rep][:, :n_pcs]
    groups = adata.obs[groupby]

    results = []
    for group in groups.unique():
        mask = (groups == group).values
        subset_high = X_high[mask]
        n_cells = subset_high.shape[0]
        row = {'Cluster': group, 'N_cells': n_cells, 'Trim': trim, 'Min_cells': min_cells,
               'l2': np.nan, **{f'tau{o}': np.nan for o in range(3, max_order + 1)}}

        if n_cells < min_cells:
            results.append({**row, 'Note': 'n < min_cells'})
            continue

        mean_pos, vec_high = _calc_high_dim_vector(subset_high, method=method,
                                                   trim=trim, min_cells=min_cells)
        mag = np.linalg.norm(vec_high)
        if not np.isfinite(mag) or mag < 1e-6:
            results.append({**row, 'Note': 'zero vector'})
            continue

        proj = (subset_high - mean_pos) @ (vec_high / mag)   # 1次元に射影
        lm = _calc_l_moments(proj, max_order=max_order, trim=trim, min_cells=min_cells)
        row['l2'] = float(lm['lambda'][1])
        for order in range(3, max_order + 1):
            row[f'tau{order}'] = float(lm['tau'][order])
        results.append({**row, 'Note': 'Success'})

    return pd.DataFrame(results)
########

def calc_ltrail_significance_table(adata,
                                   groupby: str,
                                   use_rep: str = 'X_pca',
                                   n_pcs: int = 30,
                                   method: str = 'lmoment',
                                   trim: int = None,
                                   n_boot: int = 10000,
                                   min_cells: int = None,
                                   random_state: int = 34,
                                   threshold_magnitude: float = 1e-6,
                                   ) -> pd.DataFrame:

                                   """
                                   全クラスターのraw p を計算する。(多重補正のために必要)
                                   min_cells > clster cells でもP_raw = NaN, Note = 'n < min_cells'とする。

                                   Return
                                   ----------
                                   pd.DataFrame : Cluster, N_cells, Trim, Min_cells, Magnitude, P_raw, Note, Mean_pos (object), Vector (object)
                                   """

                                   trim, min_cells = _resolve_params(trim, min_cells)
                                   X_high = adata.obsm[use_rep][:, :n_pcs]
                                   groups = adata.obs[groupby]
                                   results = []

                                   for group in groups.unique():
                                       mask = (groups == group).values
                                       subset_high = X_high[mask]
                                       n_cells = subset_high.shape[0]
                                       row = {'Cluster': group,
                                              'N_cells': n_cells,
                                              'Trim': trim if method == 'lmoment' else np.nan,
                                              'Min_cells': min_cells}

                                       if n_cells < min_cells:
                                          results.append({**row,
                                                         'Magnitude': np.nan,
                                                         'P_raw': np.nan,
                                                         'Note': 'n < min_cells',
                                                         'Mean_pos': None,
                                                         'Vector': None})
                                          continue

                                       mean_pos, vec_high = _calc_high_dim_vector(subset_high, method=method, trim=trim, min_cells=min_cells)

                                       magnitude = float(np.sqrt(np.sum(vec_high ** 2)))

                                       if not np.isfinite(magnitude) or magnitude < threshold_magnitude:
                                          p_raw, note = np.nan, 'zero vector'
                                       else:
                                          p_raw = _test_significance_high_dim(subset_high,
                                                                              observed_magnitude=magnitude,
                                                                              method=method,
                                                                              n_boot=n_boot,
                                                                              random_state=random_state,
                                                                              trim=trim,
                                                                              min_cells=min_cells)
                                          note = 'Success'

                                       results.append({**row,
                                                      'Magnitude': magnitude,
                                                      'P_raw': p_raw,
                                                      'Note': note,
                                                      'Mean_pos': mean_pos,
                                                      'Vector': vec_high})
                                   return pd.DataFrame(results)



def multiple_testing(df: pd.DataFrame,
                     p_col: str = 'P_raw',
                     method: str = 'bonferroni',
                     alpha: float = 0.05,
                     ) -> pd.DataFrame:

                     """
                     add P_adj and Significant in calc_ltrail_significance_table()
                     m is not NaN P_raw row.
                       boneferroni : p_adj = min(1, P_raw * m)
                       fdr_bh : Benjamini-Hochberg
                     """
                     df = df.copy()
                     valid = df[p_col].notna()
                     m = int(valid.sum())
                     df['P_adj'] = np.nan

                     if m == 0:
                         df['Significant'] = False
                         return df

                     if method == 'bonferroni':
                         #if minimum > 1 -> minimum = 1 ...normalize
                         df.loc[valid, 'P_adj'] = np.minimum(df.loc[valid, p_col] * m, 1.0)
                     elif method == 'fdr_bh':
                         _, p_adj, _, _ = multipletests(df.loc[valid, p_col], alpha=alpha, method='fdr_bh')
                         df.loc[valid, 'P_adj'] = p_adj
                     else:
                         raise ValueError(f"Unknown method '{method}'")

                     df['Significant'] = df['P_adj'] < alpha #Nan < alpha is False
                     print(f"[multiple_testing], method ={method}, m={m}, alpha = {alpha}"
                           f"-> int{int(df['Significant'].sum())} significant clusters")
                     return df
