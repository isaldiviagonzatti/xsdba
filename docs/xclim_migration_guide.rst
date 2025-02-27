=====================
xclim Migration Guide
=====================

`xsdba` was first developed as a submodule of `xclim`_. The reason of the split was twofold

* `xsdba` grew as a heavy submodule within `xclim` with a different aim;
* To increase collaboration with other consortiums, we want more flexibility (i.e. this will break more often)

For `xclim` users, it will still be possible to import `sdba` if `xsdba` is installed.

.. code-block:: python

    from xclim import sdba

One important difference concerns units handling. Previously, automatic unit conversion was performed in the bias adjustment
methods. This included conversions units with different dimensionality in cases where the standard name in the metadata made it
unambiguous (e.g. precipitations in `kg m-2 s-1` could be converted `mm d-1` by assuming the density of liquid water). This kind
of conversion is no longer present in `xsdba`. However, if `xclim` is installed, the previous behaviour can be emulated
by specifying a units context before a computation, e.g.

.. code-block:: python

    with xclim.core.units.units.context("hydro"):
        QM = xsdba.EmpiricalQuantileMapping.train(
            ref=pr_ref,
            hist=pr_hist,
            adapt_freq_thresh="0.1 mm/d",
            jitter_under_thresh_value="0.01 mm/d",
        )
        pr_adj = QM.adjust(sim=pr_sim)

.. _xclim: https://xclim.readthedocs.io/
