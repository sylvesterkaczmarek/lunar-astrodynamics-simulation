# Revalidating the real LOLA MOON_PA grid

The actual NASA Goddard 64 pixels/degree gridline product is too large for routine CI. Revalidate it explicitly with:

```bash
python -m pip install -e .[terrain]
python scripts/download_lola_pa_shape.py --registration gridline
python scripts/validate_lola_pa_reference.py data/LDEM64_PA_gridline_202405.grd
```

The validator checks the dated product's file size, native `lon/lat/z` netCDF structure, 64 pixels/degree grid geometry, `MOON_PA_DE421` terrain metadata, and the real reference elevations recorded in `results/lola_pa_validation.json`.

The public terrain API uses metres. The Goddard GMT/netCDF product's actual `z` samples are kilometre-valued and have no `units` attribute in the downloaded file, so `load_lola_moon_pa_grd(...)` performs the product-specific conversion to metres.

Normal CI keeps an offline regression for that conversion and does not download the 703 MB-class source file.
