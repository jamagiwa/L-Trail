# lmomentの計算を順序統計量に重みをつける方式に変更
# TL-moment (DOI: 10.1016/S0167-9473(02)00250-5)に拡張、trim=0が従来のLmoment...すなわち両端のトリムを行わない

#----------
#デフォルト値 (ltrail.DEFAULT_TRIM = 1とすれば全関数に適用される)
#----------
DEFAULT_TRIM = 0 #0 → Lmoment
DEFAULT_MIN_CELLS = 30 #とりあえず

def _resolve_params(trim = None,
                    min_cells = None,
                    max_order: int=4):
  #input
  trim = DEFAULT_TRIM if trim is None else int(trim)
  min_cells = DEFAULT_MIN_CELLS if min_cells is None else int(min_cells)

  if trim < 0:
    raise ValueError('trim must be >=0')
  if min_cells < max_order + 2 * trim:
    raise ValueError(f"min_cells = {min_cells} is too small for trim = {trim}, "
                    f"max_order = {max_order} is need min_cells >= (max_order + 2 * trim).")
  return trim, min_cells

#calculate　tlmoment weight
@lru_cache(maxsize = 256)
def _tl_weights(n: int,
                r:int,
                trim: int) -> np.ndarray:
  """
  Sample TL-moment as a weighted sum of order statistics (Elamir & Seheult, 2003)
  l_r^(t) = sum_i w_i * x_(i)
  w_i = 1/(r C(n, r+2t)) * sum_{k=0}^{r-1} (-1)^k C(r-1,k) C(i-1, r+t-1-k) C(n-i, t+k)
  trim = t = 0 reduces to Hosking's unbiased sample L-moment.
  """

  if r < 1 or trim < 0:
      raise ValueError("r must be >= 1 and trim must be >= 0.")

  if n < r + 2 * trim:
      raise ValueError(f"n={n} is too small for r={r}, trim={trim} (need n >= r + 2*trim).")

  i = np.arange(1, n + 1)
  w = sum((-1) ** k * comb(r - 1, k) * comb(i - 1, r + trim - 1 - k) * comb(n - i, trim + k)
          for k in range(r))
  w = w / (r * comb(n, r + 2 * trim))
  w.setflags(write=False)
  return w


def _calc_l_moments(
                    data: np.ndarray,
                    max_order : int = 4,
                    trim: int = None,
                    min_cells: int = None,
                    lambda2_threshold: float = 1e-12
                    ) -> dict:
  """
  Trimmed L-moments calculator.
  Shared by:
  a) _calc_l_moment_vector()  -- direction vector (-lambda_3), per PC dim.

  b)calc_cluster_l_moment_diagram()  -- tau3 / tau4 shape diagnostics in dim 1

  Parameter
  ----------
　data : np.ndarray, shape (n_cells) or (n_cells, n_dims)
  max_order : int, optional (default: 4)
  trim : int, optional (default: DEFAULT_TRIM)
  0 -> L-moments (Hoking, 1990)

  Returns
  ----------
  dict:
  'n_cells' : int
  'trim'    : int
  'lambda'   : np.ndarray, shape (max_order, ) or (max_order, n_dims)
  lambda[r-1] == lambda_r(trim)
  'tau'     :dict {order: attay} for order in 3, max_order
  (tau_r = lambda_r / lambda_2; NaN where lambda_2 ~ 0)

  Symmetric trimming keeps l3(-x) = -l3(x), PC sign flips are harmless,
  """
#Input parameter check
  if max_order < 2:
    raise ValueError("max_order must be >= 2")

#data shape
  trim, min_cells = _resolve_params(trim, min_cells, max_order=max_order)
#check cell number
  n_cells = data.shape[0]
#output data shape
  out_shape = () if data.ndim == 1 else (data.shape[1],)

#data check
  if n_cells < min_cells or n_cells < max_order + 2 * trim :
    return{'n_cells': n_cells,
           'trim': trim,
           'lambda': np.full((max_order,) + out_shape, np.nan),
           'tau': {r: np.full(out_shape, np.nan) for r in range(3, max_order + 1)}}

#sorting
#for instance (3, 1, 2, 4) -> (1, 2, 3, 4)
  sorted_data = np.sort(data, axis = 0)

#core
#@ is dot product
#for instance...weight data =: (n, ), sorted_data =: (n, d)
#(n, ) @ (n, d) = (nHd, )
  lam = np.stack([_tl_weights(n_cells, r, trim) @ sorted_data
                 for r in range(1, max_order + 1)], axis=0)

  lambda2 = lam[1]

  tau = {}

  with np.errstate(divide = 'ignore', invalid = 'ignore'):
    for order in range(3, max_order + 1):
      #tau_r = lambda_r / lambda2
      tau[order] = np.where(np.abs(lambda2) > lambda2_threshold, lam[order - 1] / lambda2, np.nan)

  return {'n_cells': n_cells,
          'trim': trim,
          'lambda': lam,
          'tau': tau}

def _calc_l_moment_vector(
                          data: np.ndarray,
                          trim: int = None,
                          min_cells: int = None
                          ) ->np.ndarray:
  """
  Calclate vector -lambda_3^(trim) for each column (default PC dimension).
  """
  trim, min_cells = _resolve_params(trim, min_cells)
  n_cells = data.shape[0]

  if n_cells < min_cells or n_cells < 3 + 2 * trim:
    return np.full(data.shape[1:], np.nan)
  return -(_tl_weights(n_cells, 3, trim) @ np.sort(data, axis=0))

def _calc_high_dim_vector(
                          data: np.ndarray,
                          method: str = 'lmoment',
                          scale_skew_std: bool = True,
                          trim: int = None,
                          min_cells: int = None,
                          ) -> tuple:


                          """
                          Calculate the centroid and directional vector for a given cluster in high-dimentional space.
                          Parameters
                          ----------
                          data : np.ndarray
                          High-dimentional data array for a specific cluster. (n_cells, n_dimen)
                          method : str , optional (default : 'lmoment')
                          - 'lmoment' : Vector based on (trimed) L-moments.
                          - 'pearson' : 3 * (median - mean). (Not divided by SD)
                          - 'skew'    : Fisher-Pearson coefficient of skewness.
                          scale_skew_std : boor, optional (default = True)
                          Only applicable when `method = 'skew'`.
                          trim : int, optional (default = DEFAULT_TRIM)
                          TL-momnet triming per side. Only use when `method = 'lmoment'`

                          Return
                          ----------
                          mean_pos : np.ndarray
                          vec : np.ndarray
                          """

                          mean_pos = np.mean(data, axis=0)

                          if method == 'pearson':
                            median_pos = np.median(data, axis=0)
                            vec = 3 * (median_pos - mean_pos)

                          elif method == 'skew':
                            sk = skew(data, axis=0)
                            if scale_skew_std:
                              vec = -sk * np.std(data, axis=0)
                            else:
                              vec = -sk

                          elif method == 'lmoment':
                            vec = _calc_l_moment_vector(data, trim=trim, min_cells=min_cells)

                          else:
                            raise ValueError(f"Unknown method '{method}'. Expected 'lmoment', 'pearson', 'skew'.")

                          return mean_pos, vec

def _test_significance_high_dim(
    subset: np.ndarray,
    observed_magnitude: float,
    method: str,
    n_boot: int = 10000,
    random_state: int = 34,
    trim: int = None,
    min_cells : int = None,
    scale_skew_std = True,
    ) -> float:
    """
    Sign-flip permutation test for norm of the directional vector.

    Parameter
    ----------
    subset : np.ndarray, shape (n_cells, n_dimensions)
    observed_magnitude : float
    method : str
    n_boot : int, optional (default - 10000)
    Number of sign-flip permutation.
    random_state : int, optional (default = 34)
    trim : int, optional (default : DEFAULT_TRIM)
    Only used when `method = 'lmoment'`

    Return
    ----------
    p_value : float
    """

    trim, min_cells = _resolve_params(trim, min_cells)
    n_cells = subset.shape[0]

#if min_cell < x -> _calc_l_moment_vector = None
#observed_magnitude = None -> null_mags >= None = 0 -> p_val = (0 + 1) / (n_boot + 1 ) -> False positive
    if (observed_magnitude is None or not np.isfinite(observed_magnitude)
       or n_cells < min_cells
       or (method == 'lmoment' and n_cells < 3 + 2 *trim)):
         return np.nan

    centered = subset - np.mean(subset, axis=0)
# random number generation
    rng = np.random.default_rng(random_state)
#n_boot = 10000 -> null_mags[0], null_mags[1],...null_mags[9999]
    null_mags = np.empty(n_boot)

#w3 : weight is constant -> out of for loop
    w3 = _tl_weights(n_cells, 3, trim) if method == 'lmoment' else None

    for b in range(n_boot):
#各細胞に-1, 1をランダムにかける
#(n_cells, 1)各cellのベクトル全体を反転している。
      sings = rng.choice([-1, 1], size=(n_cells, 1))
#shuffled
      shuffled = centered * sings
      if w3 is not None:
        vec_null = -(w3 @ np.sort(shuffled, axis=0))
      else:
        _, vec_null = _calc_high_dim_vector(shuffled, method=method, scale_skew_std=scale_skew_std)

      null_mags[b] = np.sqrt(np.sum(vec_null ** 2))

    null_mags = null_mags[np.isfinite(null_mags)]
    if null_mags.size == 0:
      return np.nan
    return (np.sum(null_mags >= observed_magnitude) + 1) / (null_mags.size + 1)
