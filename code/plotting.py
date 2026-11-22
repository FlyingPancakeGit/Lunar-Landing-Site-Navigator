from settings import *

from site_eval import evaluate_site, SITES, SUN_MIN_EL, EARTH_MIN_EL

import numpy as np
import matplotlib.pyplot as plt
from settings import *
from astroquery.jplhorizons import Horizons

SITES = {
    "Apollo 11": (0.67416, 23.47314, -1926),
    "Apollo 12": (-3.0128, -23.4219, -1422),
    "Apollo 14": (-3.64589, -17.47194, -1062),
    "Apollo 15": (26.13239, 3.6333, -1931),
    "Apollo 16": (-8.9734, 15.5011, 8),
    "Apollo 17": (20.1911, 30.7655, -2626),
    "Chang'e 3": (44.1214, 340.4884, -2640),
    "Chang'e 4": (-45.4561, 177.5885, -5927),
    "Chandrayaan-3": (-69.3741, 32.32, 532),
    "Reiner Gamma": (7.39, 301, 0.0),
    "Mare Crisium": (18.56, 61.81, -3570),
    "Schrodinger Basin": (-75.0, 132.4, -4700),
    "Mons Mouton": (-84.7906, 29.1957, 7026),
}

SUN_MIN_EL = 2.0
EARTH_MIN_EL = 5.0


def _query(body_id, lat, lon, elev_m, start, stop, step):
    elev_km = float(np.clip(elev_m / 1000.0, -9.5, 11.0))
    epochs = {"start": start, "stop": stop, "step": step}
    for el in (elev_km, 0.0):
        site = {"lon": float(lon), "lat": float(lat), "elevation": el, "body": 301}
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
    print(hdr)
    for name, r in rows:
        print(f"{name:24s} {100*r['sunlit_frac']:8.1f} {100*r['earth_vis_frac']:7.1f} "
              f"{100*r['both_frac']:6.1f} {r['max_sun_el']:7.1f} {r['longest_dark_h']:11.1f}")
    return dict(rows)


def plot_site(r, title="LUNAR ENVIRONMENT ANALYSIS", save="lunar_site_analysis.png"):
    hours = (r["jd"] - r["jd"][0]) * 24.0
    span = hours[-1]
    if span > 72:
        x, xlabel, tick_step = hours / 24.0, "Time (days from start)", max(1, int(span / 24 // 15))
        xmax = span / 24.0
    else:
        x, xlabel, tick_step = hours, "Time (hours from start)", max(1, int(span // 12))
        xmax = span

    fig, axes = plt.subplots(2, 1, figsize=(4.8, 3.0), sharex=True)
    fig.patch.set_facecolor("#101c30")

    series = [
        (axes[0], r["sun_el"], "#fbbf24", "Sun", "SUN ELEVATION", SUN_MIN_EL),
        (axes[1], r["earth_el"], "#60a5fa", "Earth", "EARTH ELEVATION", EARTH_MIN_EL),
    ]
    for ax, y, color, label, ttl, thresh in series:
        ax.set_facecolor("#17263d")
        ax.axhline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.8, label="Horizon (0°)")
        ax.axhline(thresh, color=color, linestyle=":", linewidth=0.7, alpha=0.7,
                   label=f"Thresh ({thresh:g}°)")
        ax.axhspan(0, 90, color="green", alpha=0.07)
        ax.axhspan(-90, 0, color="red", alpha=0.07)
        ax.plot(x, y, color=color, linewidth=1.2, label=label)
        lo, hi = min(-10, float(np.min(y)) - 5), max(10, float(np.max(y)) + 5)
        ax.set_ylim(lo, hi)
        ax.set_xlim(0, xmax)
        ax.grid(True, alpha=0.2)
        ax.set_title(ttl, color="white", fontsize=8, fontweight="bold", pad=2)
        ax.set_ylabel("El (°)", color="white", fontsize=7)
        ax.tick_params(colors="white", labelsize=6)
        for s in ax.spines.values():
            s.set_color("#526178")
        ax.legend(facecolor="#17263d", labelcolor="white", loc="upper right", fontsize=5.5)

    both = (r["sun_el"] > SUN_MIN_EL) & (r["earth_el"] > EARTH_MIN_EL)
    for ax in axes:
        ax.fill_between(x, ax.get_ylim()[0], ax.get_ylim()[1], where=both,
                        color="#34d399", alpha=0.12, linewidth=0, label=None)

    axes[1].set_xlabel(xlabel, color="white", fontsize=7, labelpad=2)
    fig.suptitle(f"{title} | {r['lat']:+.2f}°, {r['lon']:+.2f}°, {r['elev_m']:.0f}m",
                 color="white", fontsize=9, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.14, hspace=0.28)
    if save:
        fig.savefig(save, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig


def _style(ax):
    ax.set_facecolor("#17263d")
    ax.axhline(0, color="white", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.grid(True, alpha=0.2)
    ax.tick_params(colors="white", labelsize=6)
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    for s in ax.spines.values():
        s.set_color("#526178")


def _xaxis(r):
    hours = (r["jd"] - r["jd"][0]) * 24.0
    if hours[-1] > 72:
        return hours / 24.0, "Days"
    return hours, "Hours"


def plot_sites(results, mode="overlay", title="LUNAR SITE COMPARISON", save="lunar_sites_comparison.png"):
    names = list(results)
    cmap = plt.get_cmap("tab10")
    colors = {n: cmap(i % 10) for i, n in enumerate(names)}

    n_sites = len(names)
    fig, axes = plt.subplots(2, n_sites, figsize=(max(2.5, 1.4 * n_sites), 2.7),
                             sharex=True, sharey="row", squeeze=False)
    
    for j, n in enumerate(names):
        r = results[n]
        x, xlabel = _xaxis(r)
        both = (r["sun_el"] > SUN_MIN_EL) & (r["earth_el"] > EARTH_MIN_EL)
        
        for i, (key, col, thr) in enumerate((("sun_el", "#fbbf24", SUN_MIN_EL),
                                             ("earth_el", "#60a5fa", EARTH_MIN_EL))):
            ax = axes[i][j]
            _style(ax)
            ax.axhline(thr, color=col, linestyle=":", linewidth=0.6, alpha=0.6)
            ax.plot(x, r[key], color=col, linewidth=1.1)
            ax.fill_between(x, 0, 1, where=both, color="#34d399", alpha=0.15, linewidth=0,
                            transform=ax.get_xaxis_transform())
            lo_hi = [(np.min(results[m][key]), np.max(results[m][key])) for m in names]
            ax.set_ylim(min(-10, min(a for a, _ in lo_hi) - 5), max(10, max(b for _, b in lo_hi) + 5))
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        axes[0][j].set_title(f"{n}\n{100*np.mean(both):.0f}% usable", color="white",
                             fontsize=6.5, fontweight="bold", pad=2)
        axes[1][j].set_xlabel(xlabel, fontsize=6, labelpad=1)
    axes[0][0].set_ylabel("Sun el (°)", fontsize=6.5)
    axes[1][0].set_ylabel("Earth el (°)", fontsize=6.5)

    fig.patch.set_facecolor("#101c30")
    fig.suptitle(title, color="white", fontsize=9, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.14, wspace=0.12, hspace=0.18)
    if save:
        fig.savefig(save, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig


def plot_horizon_view(r, index=0,
                      save="LunarLandingSiteNavigator3/graphics/horizon_view.png"):

    index = max(0, min(int(index), len(r["jd"]) - 1))

    # Get the selected time's position
    sun_el = float(r["sun_el"][index])
    sun_az = float(r["sun_az"][index])

    earth_el = float(r["earth_el"][index])
    earth_az = float(r["earth_az"][index])

    fig, ax = plt.subplots(figsize=(6.4, 3.2))

    fig.patch.set_facecolor("#101c30")
    ax.set_facecolor("#17263d")

    # Horizon
    ax.axhline(0, color="white", linewidth=1.5, linestyle="--",
               label="Lunar horizon")

    # Sky and below-horizon areas
    ax.axhspan(0, 90, color="#17263d", alpha=0.5)
    ax.axhspan(-10, 0, color="#080d16", alpha=0.9)

    # Sun marker
    if sun_el >= 0:
        ax.scatter(sun_az, sun_el, color="#fbbf24",
                   s=100, marker="o", edgecolors="white",
                   linewidths=0.7, label="Sun", zorder=5)

        ax.annotate("SUN", (sun_az, sun_el),
                    xytext=(7, 8), textcoords="offset points",
                    color="#fbbf24", fontsize=9)

    # Earth marker
    if earth_el >= 0:
        ax.scatter(earth_az, earth_el, color="#60a5fa",
                   s=100, marker="o", edgecolors="white",
                   linewidths=0.7, label="Earth", zorder=5)

        ax.annotate("EARTH", (earth_az, earth_el),
                    xytext=(7, 8), textcoords="offset points",
                    color="#60a5fa", fontsize=9)

    # Indicate objects below the horizon
    if sun_el < 0:
        ax.text(0.02, 0.92, f"Sun below horizon ({sun_el:.1f}°)",
                transform=ax.transAxes, color="#fbbf24", fontsize=8)

    if earth_el < 0:
        ax.text(0.02, 0.84, f"Earth below horizon ({earth_el:.1f}°)",
                transform=ax.transAxes, color="#60a5fa", fontsize=8)

    # Compass direction labels
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.set_xticklabels(
        ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"],
        color="white"
    )

    ax.set_ylim(-10, 90)
    ax.set_yticks([-10, 0, 30, 60, 90])
    ax.set_ylabel("Elevation above horizon (°)", color="white")

    ax.set_xlabel("Azimuth / compass direction", color="white")

    ax.tick_params(colors="white", labelsize=8)
    ax.grid(color="white", alpha=0.2)

    for spine in ax.spines.values():
        spine.set_color("#526178")

    ax.set_title(
        f"LUNAR HORIZON VIEW | {r['lat']:.2f}°, {r['lon']:.2f}°",
        color="white",
        fontsize=10,
        fontweight="bold"
    )

    ax.legend(facecolor="#17263d", labelcolor="white",
              loc="upper right", fontsize=8)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150, facecolor=fig.get_facecolor())

    return fig
