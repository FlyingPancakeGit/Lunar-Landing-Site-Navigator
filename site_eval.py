from settings import *
from astroquery.jplhorizons import Horizons
from pyproj import Transformer
import rasterio


# LOLA Data
DEM_URL = "/vsicurl/https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"

def get_lunar_elevation(lat: float, lon: float) -> float:
    with rasterio.open(DEM_URL) as dataset:
        # Transform lon, lat on Moon sphere to the raster values
        lunar_geog = "+proj=longlat +R=1737400 +no_defs"
        transformer = Transformer.from_crs(lunar_geog, dataset.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)

        # Sample the single pixel at x, y via HTTP request
        for val in dataset.sample([(x, y)]):
            dn = float(val[0])

            scale = dataset.scales[0]
            offset = dataset.offsets[0]

            return dn * scale + offset

# Approximate public coordinates (deg N, deg E, m above reference sphere).
SITES = {
    "Apollo 11":     (0.67416,   23.47314, -1926),
    "Apollo 12":     (-3.0128,  -23.4219, -1422),
    "Apollo 14":     (-3.64589,  -17.47194,  -1062),
    "Apollo 15":     (26.13239,    3.6333,  -1931),
    "Apollo 16":     (-8.9734,   15.5011,  8),
    "Apollo 17":     (20.1911,   30.7655,  -2626),
    "Chang'e 3": (44.1214, 340.4884, -2640),
    "Chang'e 4": (-45.4561, 177.5885, -5927),
    "Chandrayaan-3":  (-69.3741, 32.32, 532),
    "Reiner Gamma":  (7.39, 301, 0.0),
    "Mare Crisium":  (18.56, 61.81, -3570),
    "Schrodinger Basin":  (-75.0, 132.4, -4700),
    "Mons Mouton":  (-84.7906, 29.1957, 7026),
}

SUN_MIN_EL = 2.0     # deg above horizon counted as "usable sunlight"
EARTH_MIN_EL = 5.0   # deg above horizon counted as "usable Earth link"


def _query(body_id, lat, lon, elev_m, start, stop, step):
    elev_km = float(np.clip(elev_m / 1000.0, -9.5, 11.0))   
    epochs = {"start": start, "stop": stop, "step": step}
    for el in (elev_km, 0.0):                              
        # JPL Horizons requires lunar longitude in the 0 to 360 degree range.
        # The GUI and map use -180..180, so convert here
        horizons_lon = float(lon) % 360.0
        site = {"lon": horizons_lon, "lat": float(lat), "elevation": el, "body": 301}
        try:
            return Horizons(id=body_id, location=site, epochs=epochs).ephemerides(
                quantities="4,20,21", cache=False)
        except ValueError as e:
            print(f"Horizons rejected site {site}: {e}")
            if el == 0.0:
                raise


def _windows(jd, mask):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = jd[i]
        if not m and start is not None:
            out.append((start, jd[i]))
            start = None
    if start is not None:
        out.append((start, jd[-1]))
    return out


def _time_to_change(jd, mask, t0):
    i = int(np.argmin(np.abs(jd - t0)))
    cur = bool(mask[i])
    for j in range(i + 1, len(jd)):
        if bool(mask[j]) != cur:
            return cur, (jd[j] - t0) * 24.0
    return cur, None


def evaluate_site(lat, lon, elev_m=0.0, start="2026-11-14", stop="2026-12-14",
                  step="30m", t_land_jd=None):
    sun = _query("10", lat, lon, elev_m, start, stop, step)
    earth = _query("399", lat, lon, elev_m, start, stop, step)

    jd = np.asarray(sun["datetime_jd"], dtype=float)
    s_el, s_az = np.asarray(sun["EL"], float), np.asarray(sun["AZ"], float)
    e_el, e_az = np.asarray(earth["EL"], float), np.asarray(earth["AZ"], float)
    t0 = jd[0] if t_land_jd is None else t_land_jd
    dt_h = (jd[1] - jd[0]) * 24.0

    sunlit = s_el > SUN_MIN_EL
    earth_up = e_el > EARTH_MIN_EL
    both = sunlit & earth_up

    sun_w, earth_w = _windows(jd, sunlit), _windows(jd, earth_up)
    lit_now, lit_change = _time_to_change(jd, sunlit, t0)
    ev_now, ev_change = _time_to_change(jd, earth_up, t0)

    power = np.clip(np.sin(np.radians(s_el)), 0, None)

    return {
        "lat": lat, "lon": lon, "elev_m": elev_m,
        "jd": jd, "sun_el": s_el, "sun_az": s_az, "earth_el": e_el, "earth_az": e_az,
        "earth_delta_au": np.asarray(earth["delta"], float),
        "earth_lighttime_min": np.asarray(earth["lighttime"], float),
        "sunlit_frac": float(sunlit.mean()),
        "earth_vis_frac": float(earth_up.mean()),
        "both_frac": float(both.mean()),
        "longest_sunlit_h": max([(b - a) * 24 for a, b in sun_w], default=0.0),
        "longest_dark_h": max([(b - a) * 24 for a, b in _windows(jd, ~sunlit)], default=0.0),
        "longest_earth_h": max([(b - a) * 24 for a, b in earth_w], default=0.0),
        "mean_power_proxy": float(power.mean()),
        "max_sun_el": float(s_el.max()),
        "sun_el_at_landing": float(s_el[int(np.argmin(np.abs(jd - t0)))]),
        "sunlit_at_landing": lit_now,
        "hours_to_sunset_or_sunrise": lit_change,
        "earth_visible_at_landing": ev_now,
        "hours_to_earthset_or_earthrise": ev_change,
        "step_hours": dt_h,
    }


def compare_sites(sites, start="2026-11-14", stop="2026-12-14", step="60m"):
    rows = []
    for s in sites:
        name, (lat, lon, el) = (s, SITES[s]) if isinstance(s, str) else (str(s), s)
        r = evaluate_site(lat, lon, el, start, stop, step)
        rows.append((name, r))
    hdr = f"{'site':24s} {'sunlit%':>8s} {'earth%':>7s} {'both%':>6s} {'maxSun':>7s} {'longDark(h)':>11s}"
    return dict(rows)