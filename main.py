"""GTA VI trophy-rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

A structural forecaster, written from scratch (standard library only, no
network, deterministic).  For each trophy it forecasts five quantiles of
the earn rate on Exophase at the case's target, as a pinball loss on the
logit scale scores them.

Model
-----
An earn rate is a ratio, holders / players_tracked, and the two sides move
for different reasons, so the median is built from the mechanism that
moves them, not from extrapolating the ratio's own slope:

* Players join the tracked base at some rate and then need time to reach
  each trophy.  A trophy's holders at hour h are the join rate times the
  integral of the time-to-earn curve, and the tracked base is the join
  rate times (h + c), where c is the base's age scale (initial players /
  join rate, read off a linear fit of players-tracked on h).  With a
  spread of delays to earn the trophy, its rate therefore follows

      r(h) = R * A(h) / (h + c),
      A(h) = (h - start) - tau * (1 - exp(-(h - start) / tau)),

  R its eventual rate among tracked players, start when its earners
  begin arriving, tau the spread of their delays.  That form starts
  quadratically, accelerates while the cohort matures, then flattens as
  the base ages -- which is what a log-linear trend in logit space gets
  wrong: it keeps rising at the recent pace and overshoots every trophy
  that is already bending over.  R, start and tau are fitted per trophy
  by a grid search over (start, tau) (log-space least squares for R
  given them), on the case's recent reads (last 24 h, or all of it when
  shorter).  The median blends that fit with the last rate, more so when
  the fit is well identified (many positive reads, small residual).

* A late-trophy floor: a trophy still rare at the checkpoint (under 1%)
  is one players have barely reached yet, so its rate grows faster in
  relative terms than an early trophy's.  The median is floored at a
  tier multiple of the last rate (platinum 8x, gold 4x, silver 2.5x,
  bronze 1.5x).  It is a floor, not a target, and the tier cap still
  binds above it.

* Guardrails: a tier cap (week-one ceilings: bronze 92, silver 60, gold
  35, platinum 10), a floor of 5% of the last rate and a ceiling of 15x
  it (60x for a trophy under 0.5%, whose relative moves are large).  The
  trophy every tracked player holds (rate 100%) is pinned near 100% with
  a narrow spread.

Spread
------
The logit spread sigma is built from honest, observable pieces:

* a random-walk term, 0.5 * sqrt(horizon in days) (the house baseline's);
* the residual scatter of the trophy's own cohort fit;
* the refresh-step term: the pooled size of the logit jumps a rate makes
  when the players-tracked count changes between reads.  Exophase
  refreshes a game's counts in steps, hours apart, and a refresh dilutes
  every rate new players have not earned yet.  When the history shows no
  refresh at all (the practice cases: every read identical), no step size
  can be measured, so a prior of 0.8 logit units stands in -- a refresh
  the case has not seen is exactly the risk the practice is built around;
* an extrapolation term, a quarter of the logit distance the median moved
  from the last rate.

Flat history is a special case.  Exophase's page only moves when it
refreshes, so when the case's recent reads show no refresh at all, the
predictive distribution is a mixture, not a bell: with probability 1 - p
the page is not refreshed before the target and the rate stays where it
is, with probability p one refresh lands and every rate new players have
not earned yet is diluted down by delta logit units.  The quantiles then
come from that mixture (fat lower tail, thin upper tail) instead of the
symmetric spread above.  p follows from the horizon and the refresh gap
(4.5 h when no refresh was ever seen); delta is the median refresh step
measured in the history, or, when none was ever seen, the dilution of a
refresh that multiplies the tracked base by ~6 while adding ~2 new
holders per trophy.

Everything is deterministic: plain float arithmetic, no randomness, no
clock reads, so two runs on one case write identical bytes.

Input / output follow the contest spec: read every *.in in
$MAKEFUN_INPUT_DIR (else /input), write <id>.out to $MAKEFUN_OUTPUT_DIR
(else /output) as {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Week-one ceilings per tier, in percent: a rate cannot plausibly pass these
# seven days after launch, however steep its recent slope looks.
TIER_CAP = {"platinum": 10.0, "gold": 35.0, "silver": 60.0, "bronze": 92.0}

# Late-trophy floor: how many times its last rate a still-rare trophy's
# median may at least reach, per tier.
LATE_FLOOR = {"platinum": 8.0, "gold": 4.0, "silver": 2.5, "bronze": 1.5}
LATE_BELOW = 2.0

# Fallback spread when a history shows no refresh at all: the risk of one
# the case has not seen, in logit units.
NO_REFRESH_PRIOR = 0.8

# How far the median may move from the last read, as a guardrail.
FLOOR_FRAC = 0.05
CEIL_MULT = 15.0
CEIL_MULT_RARE = 60.0

# Release day starts at 2026-11-19T00:00:00Z; the case's game block carries
# the release date, so the origin is taken from there when present.
DEFAULT_RELEASE = "2026-11-19"


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (the house baseline's rounding)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    if value > 100.0:
        value = 100.0
    if value < 0.0:
        value = 0.0
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    """ISO 8601 UTC timestamp -> epoch seconds (fractional kept)."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    from datetime import datetime

    return datetime.fromisoformat(s).timestamp()


def ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_sd, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        resid = math.sqrt(sum((y - my) ** 2 for y in ys) / n) if n else 0.0
        return my, 0.0, resid, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid, n


def median(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def cohort_fit(hours, rates, c):
    """Fit r(h) = R * A(h) / (h + c) to positive rates, grid over start/tau.

    A(h) = (h - start) - tau * (1 - exp(-(h - start) / tau)) for h > start,
    else 0: the cohort integral of an exponential time-to-earn curve.
    Log-space least squares for R given (start, tau); returns
    (R, start, tau, resid_sd, n) of the best pair, or None when the series
    cannot support a fit.
    """
    pts = [(h, r) for h, r in zip(hours, rates) if r is not None and r > 0]
    if len(pts) < 3:
        return None
    if max(r for _, r in pts) / min(r for _, r in pts) < 1.05:
        return None  # effectively flat: no slope to fit
    # Subsample to at most ~48 points: the reads are dense and the fit
    # does not need every 10-minute step.
    if len(pts) > 48:
        step = len(pts) / 48.0
        pts = [pts[int(k * step)] for k in range(48)]
    logr = [math.log(r) for _, r in pts]
    best = None
    for tau in (4.0, 8.0, 16.0, 32.0, 64.0):
        start = 0.0
        while start <= 96.0:
            vals = []
            ok = True
            for (h, _), lr in zip(pts, logr):
                d = h - start
                if d <= 0:
                    ok = False
                    break
                a = d - tau * (1.0 - math.exp(-d / tau))
                g = a / (h + c)
                if g <= 0:
                    ok = False
                    break
                vals.append(lr - math.log(g))
            if ok and vals:
                log_r = sum(vals) / len(vals)
                resid = math.sqrt(sum((v - log_r) ** 2 for v in vals) / len(vals))
                if best is None or resid < best[3]:
                    best = (math.exp(log_r), start, tau, resid, len(vals))
            start += 4.0
    return best


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0

    release = (case.get("game") or {}).get("release_date", DEFAULT_RELEASE)
    t0 = parse_time(release + "T00:00:00Z")
    t_target = parse_time(case["target"])
    h_target = max(0.0, (t_target - t0) / 3600.0)

    # --- Build the per-read series -------------------------------------
    # reads: (hours since release, players tracked, [rates...])
    reads = []
    for h in history:
        try:
            t = parse_time(h["at"])
        except (KeyError, ValueError):
            continue
        rates = h.get("rates") or []
        if len(rates) < len(trophies):
            rates = list(rates) + [None] * (len(trophies) - len(rates))
        reads.append(((t - t0) / 3600.0, h.get("players_tracked"), rates))
    reads.sort(key=lambda r: r[0])
    if not reads:
        # No history at all: a flat, wide prior per tier.
        return {"trophies": [
            {"id": t["id"], "q": quantiles_for(1.0 if t["tier"] != "platinum" else 0.5, 1.0)}
            for t in trophies
        ]}

    h_last = reads[-1][0]
    # Recent window: the last 24 h of reads; everything when shorter.
    window = [r for r in reads if r[0] >= h_last - 24.0]
    if len(window) < 3:
        window = reads

    # --- Players-tracked growth ----------------------------------------
    # Linear fit N = N0 + J*h on the window: the base's age scale c = N0/J.
    npts = [(h, n) for (h, n, _) in window if n and n > 0]
    if len(npts) >= 2:
        a_n, b_n, _, _ = ols([p[0] for p in npts], [p[1] for p in npts])
    elif npts:
        a_n, b_n = npts[0][1], 0.0
    else:
        a_n, b_n = 638.0, 0.0
    if b_n > 0 and a_n >= 0:
        c_age = min(48.0, max(0.0, a_n / b_n))
    else:
        c_age = 2.0
    n_last = reads[-1][1] or 638

    # --- Refresh-step size ---------------------------------------------
    # Pooled |logit jump| of a rate across consecutive reads where the
    # players-tracked count changed: the size of one Exophase refresh step.
    steps = []
    for (h0, n0, r0), (h1, n1, r1) in zip(reads, reads[1:]):
        if n0 and n1 and n1 != n0:
            for a, b in zip(r0, r1):
                if a is not None and b is not None and a > 0 and b > 0:
                    steps.append(abs(logit(b) - logit(a)))
    sigma_step = median(steps) if steps else NO_REFRESH_PRIOR
    sigma_step = min(1.0, max(0.0, sigma_step))

    # Is the recent window flat -- no refresh observed in it?
    window_flat = True
    for (_, n0, _), (_, n1, _) in zip(window, window[1:]):
        if n0 and n1 and n0 != n1:
            window_flat = False
            break
    # Refresh gap, for the flat-window mixture: from the full history when
    # refreshes were seen there, else the 4.5 h prior.
    n_changes = sum(
        1
        for (_, n0, _), (_, n1, _) in zip(reads, reads[1:])
        if n0 and n1 and n0 != n1
    )
    if n_changes >= 1 and h_last > reads[0][0]:
        gap_hours = (h_last - reads[0][0]) / n_changes
        gap_hours = min(12.0, max(1.0, gap_hours))
    else:
        gap_hours = 4.5

    win_hours = [r[0] for r in window]

    rows = []
    for i, t in enumerate(trophies):
        # Last non-null rate.
        last_rate = None
        for (_, _, rates) in reversed(reads):
            if rates[i] is not None:
                last_rate = rates[i]
                break
        if last_rate is None:
            last_rate = 1.0

        # The trophy everybody holds stays near everybody holding it.
        if last_rate >= 99.99:
            rows.append({"id": t["id"], "q": quantiles_for(99.9, 0.15)})
            continue

        # Flat recent window, short horizon: the refresh mixture.  The
        # page only moves at a refresh, so the predictive distribution is
        # two branches -- no refresh (rate stays) vs one refresh (rate
        # diluted by delta) -- not a symmetric bell.
        if window_flat and case["horizon_minutes"] <= 12 * 60 and last_rate > 0:
            p_refresh = 1.0 - math.exp(-(case["horizon_minutes"] / 60.0) / gap_hours)
            if steps:
                delta = min(2.5, max(0.8, median(steps)))
            else:
                holders_last = last_rate * n_last / 100.0
                frac = (holders_last + 2.0) / (6.0 * holders_last)
                delta = min(2.5, max(0.8, -math.log(frac)))
            rows.append({"id": t["id"], "q": mixture_quantiles(last_rate, p_refresh, delta)})
            continue

        # Cohort fit for the median.
        win_rates = [rates[i] for (_, _, rates) in window]
        fit = cohort_fit(win_hours, win_rates, c_age)
        if fit is not None:
            r_inf, start_fit, tau_fit, resid, n_fit = fit
            d_t = h_target - start_fit
            a_t = d_t - tau_fit * (1.0 - math.exp(-d_t / tau_fit)) if d_t > 0 else 0.0
            m_cohort = r_inf * a_t / (h_target + c_age) if a_t > 0 else 0.01
            if n_fit >= 10 and resid < 0.25:
                w = 0.9
            elif n_fit >= 10 and resid < 0.4:
                w = 0.8
            elif n_fit >= 5 and resid < 0.7:
                w = 0.6
            else:
                w = 0.35
            med_logit = w * logit(m_cohort) + (1.0 - w) * logit(last_rate)
            med = inv_logit(med_logit)
            sigma_fit = min(0.8, max(0.05, resid))
            move = abs(logit(m_cohort) - logit(last_rate))
        else:
            med = last_rate
            sigma_fit = 0.3
            move = 0.0

        # Late-trophy floor: a still-rare trophy grows faster, relatively.
        if 0 < last_rate < LATE_BELOW:
            med = max(med, last_rate * LATE_FLOOR.get(t["tier"], 1.5))

        # Guardrails.
        cap = TIER_CAP.get(t["tier"], 92.0)
        if last_rate > 0:
            med = min(cap, max(last_rate * FLOOR_FRAC, med))
            ceil = last_rate * (CEIL_MULT_RARE if last_rate < 0.5 else CEIL_MULT)
            med = min(ceil, max(med, 0.01))
        else:
            med = min(cap, max(0.005, med))
            med = min(5.0, med)
        med = clamp_pct(med)

        # Spread: random walk + fit residual + refresh step + move size.
        sigma_rw = 0.5 * math.sqrt(horizon_days)
        sigma_extrap = min(0.5, 0.25 * move)
        sigma = math.sqrt(sigma_rw ** 2 + sigma_fit ** 2 + sigma_step ** 2 + sigma_extrap ** 2)
        sigma = min(1.6, max(0.15, sigma))

        rows.append({"id": t["id"], "q": quantiles_for(med, sigma)})

    return {"trophies": rows}


def quantiles_for(med, sigma):
    lg = logit(med)
    out = []
    prev = None
    for z in Z:
        q = round4(inv_logit(lg + z * sigma))
        q = min(100.0, max(0.01, q))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mixture_quantiles(med, p_refresh, delta, sigma0=0.15, sigma1=0.5):
    """Quantiles of the flat-history refresh mixture, in logit space.

    With probability 1 - p_refresh the page is not refreshed before the
    target and the rate stays at its median (spread sigma0); with
    probability p_refresh one refresh lands and the logit rate drops by
    delta (spread sigma1).  Each level is inverted from the mixture CDF by
    bisection; deterministic.
    """
    lg = logit(med)

    def cdf(x):
        return (1.0 - p_refresh) * _phi(x / sigma0) + p_refresh * _phi((x + delta) / sigma1)

    out = []
    prev = None
    for tau in LEVELS:
        lo, hi = -8.0, 4.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if cdf(mid) < tau:
                lo = mid
            else:
                hi = mid
        q = round4(inv_logit(lg + 0.5 * (lo + hi)))
        q = min(100.0, max(0.01, q))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
