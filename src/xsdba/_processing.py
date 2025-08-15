"""
# noqa: SS01
Compute Functions Submodule
===========================

Here are defined the functions wrapped by map_blocks or map_groups.
The user-facing, metadata-handling functions should be defined in processing.py.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

import xsdba
from xsdba import nbutils as nbu
from xsdba.base import Grouper, map_groups
from xsdba.utils import ADDITIVE, apply_correction, ecdf, get_correction, invert, rank


@map_groups(
    sim_ad=[Grouper.ADD_DIMS, Grouper.DIM],
    dP0=[Grouper.ADD_DIMS, Grouper.PROP],
    P0_ref=[Grouper.ADD_DIMS, Grouper.PROP],
    P0_hist=[Grouper.ADD_DIMS, Grouper.PROP],
    pth=[Grouper.ADD_DIMS, Grouper.PROP],
)
def _adapt_freq(
    ds: xr.Dataset, *, dim: Sequence[str], thresh: float = 0, kind: str = "+"
) -> xr.Dataset:
    r"""
    Adapt frequency of values under thresh of `sim`, in order to match ref.

    This is the compute function, see :py:func:`xsdba.processing.adapt_freq` for the user-facing function.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset variables:
            sim : simulated data
            ref (optional) : training target.
            P0_ref (optional) : Proportion of zeros in the reference dataset.
            P0_hist (optional) : Proportion of zeros in the historical dataset.
            pth (optional) : The smallest value of sim that was not frequency-adjusted in the training.
    dim : str, or sequence of strings
        Dimension name(s).
        If more than one, the probabilities and quantiles are computed within all the dimensions.
        If `window` is in the names, it is removed before the correction and the final timeseries is corrected along dim[0] only.
    group : Union[str, Grouper]
        Grouping information, see base.Grouper.
    thresh : float
        Threshold below which values are considered zero.

    Returns
    -------
    xr.Dataset, with the following variables:
      - `sim_adj`: Simulated data with the same frequency of values under threshold than ref.
        Adjustment is made group-wise.
      - `pth` : For each group, the smallest value of sim that was not frequency-adjusted.
        All values smaller were either left as zero values or given a random value between thresh and pth.
        NaN where frequency adaptation wasn't needed.
      - `P0_ref` : For each group, the percentage of values under threshold in ref.
      - `P0_hist` : For each group, either the percentage of values under threshold in sim,
        or simply repeating the input `ds.P0_hist`.

    Notes
    -----
        `ds.ref` is optional: If `P0_ref`, `P0_hist`,`pth` are given, these values will be used and `ds.ref` is not necessary.
        Either `ds.ref` or the triplet (`P0_ref`, `P0_hist`,`pth`)  must be given.
    """
    ref, P0_ref, P0_hist, pth = (
        ds.get(k, None) for k in ["ref", "P0_ref", "P0_hist", "pth"]
    )
    reuse_adapt_output = {P0_ref is not None, P0_hist is not None, pth is not None}
    if len(reuse_adapt_output) != 1:
        raise ValueError("`P0_ref`, `P0_hist`, `pth` must all be given, or be `None`.")
    reuse_adapt_output = list(reuse_adapt_output)[0]
    if len({ref is not None, reuse_adapt_output}) != 2:
        raise ValueError(
            "Either `ref` or the triplet (`P0_ref`,`P0_hist`,`pth`) must be None."
        )

    # only reduce along specified dimensions to avoid collapsing ensemble/spatial axes
    dim = [dim] if isinstance(dim, str) else list(dim)
    # map_groups quirk: datasets are broadcasted and must be sliced

    if reuse_adapt_output:  # only slice when using precomputed values
        P0_ref, P0_hist, pth = (
            da if da is None else da[{d: 0 for d in set(dim).intersection(set(da.dims))}]
            for da in [P0_ref, P0_hist, pth]
        )

    red_sim = [d for d in dim if d in ds.sim.dims]

    # compute frequencies only along intended dimensions
    red_ref = [d for d in red_sim if ref is not None and d in ref.dims] if ref is not None else red_sim
    
    # Compute the probability of finding a value <= thresh
    # This is the "dry-day frequency" in the precipitation case
    P0_sim = ecdf(ds.sim, thresh, dim=red_sim)
    P0_hist = P0_sim if P0_hist is None else P0_hist
    P0_ref = ecdf(ref, thresh, dim=red_ref) if P0_ref is None else P0_ref

    # broadcast P0_ref to match P0_hist dimensions if needed
    if ref is not None:
        extra_dims_in_hist = [d for d in P0_hist.dims if d not in P0_ref.dims]
        if extra_dims_in_hist:
            P0_ref = P0_ref.broadcast_like(P0_hist)

    dP0 = (P0_hist - P0_ref) / P0_hist

    if dP0.isnull().all():
        pth = dP0.copy()
        sim_ad = ds.sim.copy()
    else:
        # Compute : ecdf_ref^-1( ecdf_sim( thresh ) )
        # The value in ref with the same rank as the first non-zero value in sim.
        # pth is meaningless when freq. adaptation is not needed
        if pth is None:
            extra_dims = [d for d in P0_hist.dims if d not in ref.dims]
            if extra_dims:
                # broadcast reference to match P0_hist for threshold computation
                ref_for_pth = ref.broadcast_like(P0_hist)
            else:
                #  original case
                ref_for_pth = ref
            pth = nbu.vecquantiles(ref_for_pth, P0_hist, red_ref).where(dP0 > 0)

        sim = ds.sim
        # Get the percentile rank of each value in sim.
        rnk = rank(sim, dim=red_sim, pct=True)

        # Broadcast frequency stats to match simulation shape
        P0_ref_corr = P0_ref.broadcast_like(P0_sim)
        P0_hist_corr = P0_hist.broadcast_like(P0_sim)


        sim_ad = sim.where(
            dP0 < 0,  # dP0 < 0 means no-adaptation.
            sim.where(
                (rnk < (P0_ref_corr / P0_hist_corr) * P0_sim)
                | (rnk > P0_sim)
                | sim.isnull(),  # Preserve current values
                # Generate random numbers ~ U[T0, Pth]
                (pth.broadcast_like(sim) - thresh)
                * np.random.random_sample(size=sim.shape).astype(sim.dtype)
                + thresh,
            ),
        )

    # Tell group_apply that these will need reshaping (regrouping)
    # This is needed since if any variable comes out a `groupby` with the original group axis,
    # the whole output is broadcasted back to the original dims.
    for v in ("pth", "dP0", "P0_ref", "P0_hist"):
        locals()[v].attrs["_group_apply_reshape"] = True

    return xr.Dataset(
        data_vars={
            "pth": pth,
            "dP0": dP0,
            "P0_ref": P0_ref,
            "P0_hist": P0_hist,
            "sim_ad": sim_ad,
        }
    )


@map_groups(
    reduces=[Grouper.DIM, Grouper.PROP], data=[Grouper.DIM], norm=[Grouper.PROP]
)
def _normalize(
    ds: xr.Dataset,
    *,
    dim: Sequence[str],
    kind: str = ADDITIVE,
) -> xr.Dataset:
    """
    Normalize an array by removing its mean.

    Parameters
    ----------
    ds : xr.Dataset
        The variable `data` is normalized.
        If a `norm` variable is present, is uses this one instead of computing the norm again.
    group : Union[str, Grouper]
        Grouping information. See :py:class:`xsdba.base.Grouper` for details.
    dim : sequence of strings
        Dimension name(s).
    kind : {'+', '*'}
        How to apply the adjustment, using either additive or multiplicative methods.

    Returns
    -------
    xr.Dataset
        Group-wise anomaly of x.

    Notes
    -----
    Normalization is performed group-wise.
    """
    if "norm" in ds:
        norm = ds.norm
    else:
        norm = ds.data.mean(dim=dim)
    norm.attrs["_group_apply_reshape"] = True

    return xr.Dataset(
        {"data": apply_correction(ds.data, invert(norm, kind), kind), "norm": norm}
    )


@map_groups(reordered=[Grouper.DIM], main_only=False)
def _reordering(ds: xr.Dataset, *, dim: str) -> xr.Dataset:
    """
    Group-wise reordering.

    Parameters
    ----------
    ds : xr.Dataset
        With variables:
            - sim : The timeseries to reorder.
            - ref : The timeseries whose rank to use.
    dim : str
        The dimension along which to reorder.

    Returns
    -------
    xr.Dataset
        The reordered timeseries.
    """

    def _reordering_1d(data, ordr):
        return np.sort(data)[np.argsort(np.argsort(ordr))]

    def _reordering_2d(data, ordr):
        data_r = data.ravel()
        ordr_r = ordr.ravel()
        reorder = np.sort(data_r)[np.argsort(np.argsort(ordr_r))]
        return reorder.reshape(data.shape)[
            :, int(data.shape[1] / 2)
        ]  # pick the middle of the window

    if {"window", "time"} == set(dim):
        return (
            xr.apply_ufunc(
                _reordering_2d,
                ds.sim,
                ds.ref,
                input_core_dims=[["time", "window"], ["time", "window"]],
                output_core_dims=[["time"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[ds.sim.dtype],
            )
            .rename("reordered")
            .to_dataset()
        )

    if len(dim) == 1:
        return (
            xr.apply_ufunc(
                _reordering_1d,
                ds.sim,
                ds.ref,
                input_core_dims=[dim, dim],
                output_core_dims=[dim],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[ds.sim.dtype],
            )
            .rename("reordered")
            .to_dataset()
        )

    raise ValueError(
        f"Reordering can only be done along one dimension. "
        f"If there is more than one, they should be `window` and `time`. "
        f"The dimensions are {dim}."
    )
