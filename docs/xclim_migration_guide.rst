=====================
xclim Migration Guide
=====================

`xsdba` was first developed as a submodule of `xclim`_. The reason of the split was twofold

* `xsdba` grew as a heavy submodule within `xclim` with a different aim;
* To increase collaboration with other consortiums, we want more flexibility (i.e. this will break more often)

For `xclim` users, it will still be possible to import `sdba` if `xsdba` is installed.

.. code-block:: python

    from xclim import sdba


Units handling
--------------

One important change concerns units handling.  Conversion between units with different dimensionality was previously automated, when possible, in `xclim.sdba`, given that the target dataset had a suitable standard name (e.g. allowing conversion of `lwe_precipitation_rate` from `kg m-2 s-1` to `mm d-1`). `xsdba` will not perform these automatic conversions, thus the datasets and thresholds to be compared should be given in units that share the same dimensionality before being passed to `xsdba` functions

 `xsdba` still implements some unit handling through `pint`. High level adjustment objects will (usually) parse units from the passed xarray objects, check that the different inputs have matching units and perform conversions when units are compatible but don't match. `xsdba` imports `cf-xarray's unit registry <https://cf-xarray.readthedocs.io/en/latest/units.html>`_ by default and, as such, expects units matching the `CF conventions <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#units>`_. This is very similar to `xclim`, except for two features:

    - `xclim` is able to detect more complex conversions based on the CF "standard name". `xsdba` will not implement this, the user is expected to provide coherent inputs to the adjustment objects.
    - `xclim` has a convenience "hydro" context that allows for conversion between rates and fluxes, thicknesses and amounts of liquid water quantities by using an implicit the water density (1000 kg/m³).

The context of that last point can still be used if `xclim` is imported along with `xsdba` :

.. code-block:: python

    import xsdba
    import xclim  # importing xclim makes xsdba use xclim's unit registry

    pr_ref  # reference precipitation data in kg m-2 s-1
    pr_hist  # historical precipitation data in mm/d
    # In normal xsdba, the two quantities are not compatible.
    # But with xclim's hydro context, an implicit density allows for conversion

    with xclim.core.units.units.context("hydro"):
        QDM = xsdba.QuantileDeltaMapping.train(
            ref=pr_ref,
            hist=pr_hist,
            adapt_freq_thresh="0.1 mm/d",
            jitter_under_thresh_value="0.01 mm/d",
        )
        pr_adj = QM.adjust(sim=pr_sim)
..

Under the hood, `xsdba` picks whatever unit registry has been declared the "application registry" (`see pint's doc <https://pint.readthedocs.io/en/stable/api/base.html#pint.get_application_registry>`_). However, it expects some features as declared in ``cf-xarray``, so a compatible registry (as `xclim`'s) must be used.


.. _xclim: https://xclim.readthedocs.io/
