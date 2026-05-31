"""
3_KarlWeierstrass_NEXT8_2c3b — POGODNI deo iz 1_KarlWeierstrass_v2.py
Aparat 2c: Fraktalna dimenzija  +  Test 3b: Autokorelacija (ACF)

Self-contained:
  - KORAK 1: ucitavanje 4624 izvlacenja i izgradnja f(t) = lex-indeks
  - KORAK 2c: rolling/local Higuchi FD (priprema)
  - KORAK 2c3b: ACF nad rolling FD nizom, ACF nad f(t) kao kontrola,
                Ljung-Box i shuffled max|ACF| referenca

Output:
  3_KarlWeierstrass_NEXT8_2c3b.png
  3_KarlWeierstrass_NEXT8_2c3b.txt
"""

import csv
import math
import os
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


T0 = time.time()

CSV_DRAWS = "/data/loto7_4624_k43.csv"

HERE = os.path.dirname(os.path.abspath(__file__))
PNG_PATH = os.path.join(HERE, "3_KarlWeierstrass_NEXT8_2c3b.png")
TXT_PATH = os.path.join(HERE, "3_KarlWeierstrass_NEXT8_2c3b.txt")

N_MAX = 39
K_PICK = 7
TOTAL_COMBOS = math.comb(N_MAX, K_PICK)


# ─── helperi (samo oni potrebni za 2c + 2c3b) ────────────────────────
def read_loto_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < K_PICK:
                continue
            try:
                nums = tuple(sorted(int(x) for x in row[:K_PICK]))
            except ValueError:
                continue
            if len(nums) == K_PICK and len(set(nums)) == K_PICK:
                rows.append(nums)
    return rows


def lex_rank_1based(combo, n=N_MAX, k=K_PICK):
    """1-based lex indeks (poklapa se sa rednim brojem u kombinacije_39C7.csv)."""
    combo = tuple(sorted(combo))
    rank0 = 0
    prev = 0
    for i, value in enumerate(combo):
        remaining = k - i - 1
        for candidate in range(prev + 1, value):
            rank0 += math.comb(n - candidate, remaining)
        prev = value
    return rank0 + 1


def normalize01(x):
    x = np.asarray(x, dtype=float)
    span = float(x.max() - x.min())
    return (x - x.min()) / (span + 1e-12)


def higuchi_fd(series, kmax=64):
    """Higuchi fractal dimension za 1D vremenski niz."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    ks = np.arange(1, min(kmax, n // 2) + 1, dtype=int)
    lk = []
    used = []

    for k in ks:
        lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            dist = np.abs(np.diff(x[idx])).sum()
            norm = (n - 1) / ((len(idx) - 1) * k)
            lm.append((dist * norm) / k)
        if lm:
            used.append(k)
            lk.append(float(np.mean(lm)))

    used = np.asarray(used, dtype=float)
    lk = np.asarray(lk, dtype=float)
    slope, intercept = np.polyfit(np.log(1.0 / used), np.log(lk), 1)
    fit = intercept + slope * np.log(1.0 / used)
    ss_res = float(np.sum((np.log(lk) - fit) ** 2))
    ss_tot = float(np.sum((np.log(lk) - np.log(lk).mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2), used, lk


def rolling_higuchi_fd(series, window=768, step=128, kmax=32):
    """Rolling Higuchi FD procena kroz vreme."""
    x = np.asarray(series, dtype=float)
    centers = []
    fdvals = []
    r2vals = []
    for start in range(0, len(x) - window + 1, step):
        seg = normalize01(x[start:start + window])
        fd, _, r2, _, _ = higuchi_fd(seg, kmax=min(kmax, window // 4))
        centers.append(start + window // 2 + 1)
        fdvals.append(fd)
        r2vals.append(r2)
    return (
        np.asarray(centers, dtype=float),
        np.asarray(fdvals, dtype=float),
        np.asarray(r2vals, dtype=float),
    )


def autocorr_values(series, max_lag=60):
    """ACF za lagove 0..max_lag."""
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    vals = []
    for lag in range(max_lag + 1):
        if lag == 0:
            vals.append(1.0)
        else:
            vals.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return np.asarray(vals, dtype=float)


def ljung_box_approx(acf_vals, n, h):
    """Ljung-Box Q aproksimacija za prvih h ACF lagova."""
    lags = np.arange(1, h + 1, dtype=float)
    q = n * (n + 2) * float(np.sum((acf_vals[1:h + 1] ** 2) / (n - lags)))
    p = float(stats.chi2.sf(q, h))
    return float(q), p


# ─── KORAK 1: f(t) = lex-indeks ──────────────────────────────────────
draws = read_loto_csv(CSV_DRAWS)
N = len(draws)
lex_idx = np.array([lex_rank_1based(c) for c in draws], dtype=np.float64)

print()
print("3_KarlWeierstrass_NEXT8_2c3b — KORAK 1: formiranje krive f(t)")
print(f"  CSV:                  {CSV_DRAWS}")
print(f"  Ucitano izvlacenja:    {N}")
print(f"  C(39,7):              {TOTAL_COMBOS:,}")
print()

with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.write("3_KarlWeierstrass_NEXT8_2c3b — Fraktalna dimenzija + ACF (POGODNO)\n")
    f.write("=" * 60 + "\n\n")
    f.write("KORAK 1: Weierstrass-ova funkcija nad svih izvucenih kombinacija\n\n")
    f.write(f"  CSV izvucenih:        {CSV_DRAWS}\n")
    f.write(f"  Ucitano izvlacenja:    {N}\n")
    f.write(f"  C(39,7):              {TOTAL_COMBOS:,}\n")
    f.write("  f(t) = lex-indeks izvucene kombinacije u skupu svih 39C7\n\n")


# ─── KORAK 2c: rolling/local Higuchi FD (priprema) ───────────────────
fd_roll_window = 768
fd_roll_step = 128
fd_roll_centers, fd_roll, fd_roll_r2 = rolling_higuchi_fd(
    lex_idx, window=fd_roll_window, step=fd_roll_step, kmax=32
)


# ─── KORAK 2c3b: ACF nad rolling FD + kontrola + shuffled referenca ──
T0_2C3B = time.time()

fd_acf_max_lag = min(20, max(1, len(fd_roll) - 2))
fd_acf_lags = np.arange(0, fd_acf_max_lag + 1)
acf_fd_roll = autocorr_values(fd_roll, fd_acf_max_lag)
acf_fd_control = autocorr_values(lex_idx, 60)

fd_acf_band = 1.96 / np.sqrt(len(fd_roll))
fd_acf_body = acf_fd_roll[1:]
fd_max_abs_acf = float(np.max(np.abs(fd_acf_body)))
fd_sig_lag_count = int(np.sum(np.abs(fd_acf_body) > fd_acf_band))
fd_top_idx = np.argsort(np.abs(fd_acf_body))[-min(10, len(fd_acf_body)):][::-1] + 1
fd_top_acf_pairs = [(int(lag), float(acf_fd_roll[lag])) for lag in fd_top_idx]

fd_lb_h = min(10, fd_acf_max_lag)
fd_lb_q, fd_lb_p = ljung_box_approx(acf_fd_roll, len(fd_roll), fd_lb_h)

rng_2c3b = np.random.default_rng(51)
fd_acf_shuffle_runs = 500
shuffle_fd_max_abs_acf = []
for _ in range(fd_acf_shuffle_runs):
    shuffled_fd = rng_2c3b.permutation(fd_roll)
    shuffled_acf = autocorr_values(shuffled_fd, fd_acf_max_lag)
    shuffle_fd_max_abs_acf.append(float(np.max(np.abs(shuffled_acf[1:]))))
shuffle_fd_max_abs_acf = np.asarray(shuffle_fd_max_abs_acf, dtype=float)
shuffle_fd_acf_mean = float(shuffle_fd_max_abs_acf.mean())
shuffle_fd_acf_std = float(shuffle_fd_max_abs_acf.std(ddof=1))
shuffle_fd_acf_p = float(np.mean(shuffle_fd_max_abs_acf >= fd_max_abs_acf))
shuffle_fd_acf_z = (fd_max_abs_acf - shuffle_fd_acf_mean) / (shuffle_fd_acf_std + 1e-12)

if fd_lb_p <= 0.05 or shuffle_fd_acf_p <= 0.05:
    fd_acf_note = "rolling FD ima ACF signal iznad shuffled reference"
else:
    fd_acf_note = "rolling FD nema jak ACF signal iznad shuffled reference"

print()
print("KORAK 2c3b: Aparat 2c Fraktalna dimenzija + Test 3b Autokorelacija (ACF)")
print(f"  rolling FD max |ACF| lag 1..{fd_acf_max_lag}: {fd_max_abs_acf:.4f}")
print(f"  95% band: +/-{fd_acf_band:.4f}   znacajnih lagova: "
      f"{fd_sig_lag_count}/{fd_acf_max_lag}")
print(f"  Ljung-Box aproks. h={fd_lb_h}: Q={fd_lb_q:.2f}  p={fd_lb_p:.4f}")
print(f"  shuffled max|ACF|: mean={shuffle_fd_acf_mean:.4f} std={shuffle_fd_acf_std:.4f} "
      f"z={shuffle_fd_acf_z:.2f} p={shuffle_fd_acf_p:.4f}")
print(f"  ⇒ {fd_acf_note}")
print()

fig2c3b, ax2c3b = plt.subplots(1, 3, figsize=(16, 5))
fig2c3b.suptitle("KORAK 2c3b: Fraktalna dimenzija + ACF test  (POGODNO)",
                 fontsize=13, fontweight="bold")

ax2c3b[0].bar(fd_acf_lags[1:], acf_fd_roll[1:], width=0.8, color="purple")
ax2c3b[0].axhline(fd_acf_band, color="crimson", linestyle="--", linewidth=1.2)
ax2c3b[0].axhline(-fd_acf_band, color="crimson", linestyle="--", linewidth=1.2)
ax2c3b[0].axhline(0, color="black", linewidth=0.6)
ax2c3b[0].set_title("ACF rolling/local FD niza")
ax2c3b[0].set_xlabel("lag")
ax2c3b[0].set_ylabel("ACF")

ax2c3b[1].bar(np.arange(1, 61), acf_fd_control[1:], width=0.8, color="steelblue")
ax2c3b[1].axhline(1.96 / np.sqrt(N), color="crimson", linestyle="--", linewidth=1.2)
ax2c3b[1].axhline(-1.96 / np.sqrt(N), color="crimson", linestyle="--", linewidth=1.2)
ax2c3b[1].axhline(0, color="black", linewidth=0.6)
ax2c3b[1].set_title("Kontrola: ACF f(t)")
ax2c3b[1].set_xlabel("lag")
ax2c3b[1].set_ylabel("ACF")

ax2c3b[2].hist(shuffle_fd_max_abs_acf, bins=24, color="lightgray", edgecolor="white")
ax2c3b[2].axvline(fd_max_abs_acf, color="crimson", linewidth=2,
                  label=f"observed={fd_max_abs_acf:.3f}")
ax2c3b[2].axvline(shuffle_fd_acf_mean, color="black", linestyle="--",
                  label=f"shuffle mean={shuffle_fd_acf_mean:.3f}")
ax2c3b[2].set_title("Shuffled rolling FD max |ACF|")
ax2c3b[2].set_xlabel("max |ACF|")
ax2c3b[2].set_ylabel("broj")
ax2c3b[2].legend(fontsize=8)

for a in ax2c3b:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(True, alpha=0.2)

fig2c3b.tight_layout()
fig2c3b.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
plt.show()

with open(TXT_PATH, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("KORAK 2c3b: Aparat 2c Fraktalna dimenzija + Test 3b Autokorelacija (ACF)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"  PNG:                  {PNG_PATH}\n\n")
    f.write("ACF nad rolling/local FD nizom:\n")
    f.write(f"  broj rolling FD tacaka = {len(fd_roll)}\n")
    f.write(f"  max lag               = {fd_acf_max_lag}\n")
    f.write(f"  95% band              = +/-{fd_acf_band:.6f}\n")
    f.write(f"  max |ACF|             = {fd_max_abs_acf:.6f}\n")
    f.write(f"  znacajnih lagova      = {fd_sig_lag_count}/{fd_acf_max_lag}\n")
    f.write(f"  Ljung-Box h           = {fd_lb_h}\n")
    f.write(f"  Ljung-Box Q           = {fd_lb_q:.6f}\n")
    f.write(f"  Ljung-Box p           = {fd_lb_p:.6f}\n\n")
    f.write("Shuffled rolling FD max |ACF| referenca:\n")
    f.write(f"  runs                  = {fd_acf_shuffle_runs}\n")
    f.write(f"  mean                  = {shuffle_fd_acf_mean:.6f}\n")
    f.write(f"  std                   = {shuffle_fd_acf_std:.6f}\n")
    f.write(f"  z                     = {shuffle_fd_acf_z:.6f}\n")
    f.write(f"  p(shuffled >= obs)    = {shuffle_fd_acf_p:.6f}\n")
    f.write(f"  interpret.            = {fd_acf_note}\n\n")
    f.write("Top ACF lagovi rolling FD po apsolutnoj vrednosti:\n")
    f.write(f"  {'lag':<8}{'ACF':>16}\n")
    for lag, val in fd_top_acf_pairs:
        f.write(f"  {lag:<8}{val:>16,.8f}\n")
    f.write("\n")

    elapsed_2c3b = time.time() - T0_2C3B
    f.write(f"Vreme KORAKA 2c3b: {timedelta(seconds=int(elapsed_2c3b))} ({elapsed_2c3b:.1f} s)\n")
    f.write(f"Ukupno vreme:       {timedelta(seconds=int(time.time()-T0))} ({time.time()-T0:.1f} s)\n")

print(f"PNG saved → {PNG_PATH}")
print(f"TXT saved → {TXT_PATH}")
print(f"Vreme KORAKA 2c3b: {timedelta(seconds=int(time.time()-T0_2C3B))} "
      f"({time.time()-T0_2C3B:.1f} s)")
print(f"Ukupno vreme:      {timedelta(seconds=int(time.time()-T0))} "
      f"({time.time()-T0:.1f} s)")
print()
print("KRAJ 3_KarlWeierstrass_NEXT8_2c3b.")
print()
"""
3_KarlWeierstrass_NEXT8_2c3b — KORAK 1: formiranje krive f(t)
  CSV:                  /data/loto7_4624_k43.csv
  Ucitano izvlacenja:   4624
  C(39,7):              15,380,937


KORAK 2c3b: Aparat 2c Fraktalna dimenzija + Test 3b Autokorelacija (ACF)
  rolling FD max |ACF| lag 1..20: 0.5257
  95% band: +/-0.3520   znacajnih lagova: 2/20
  Ljung-Box aproks. h=10: Q=30.45  p=0.0007
  shuffled max|ACF|: mean=0.3119 std=0.0740 z=2.89 p=0.0140
  ⇒ rolling FD ima ACF signal iznad shuffled reference

PNG saved → /3_KarlWeierstrass_NEXT8_2c3b.png
TXT saved → /3_KarlWeierstrass_NEXT8_2c3b.txt
Vreme KORAKA 2c3b: 0:00:14 (14.0 s)
Ukupno vreme:      0:00:14 (14.1 s)

KRAJ 3_KarlWeierstrass_NEXT8_2c3b.
"""



###############   PREDIKCIJA 8  ###############################

"""
NEXT8 (2c3b, rolling FD ACF) — AR(1) na FD; FD↑ veća magnituda inkrementa, FD↓ manja.
"""


def lex_unrank_1based(rank, n=N_MAX, k=K_PICK):
    """Vracanje 1-based lex indeksa u Loto 7/39 kombinaciju."""
    rank0 = int(rank) - 1
    combo = []
    prev = 0
    for i in range(k):
        remaining = k - i - 1
        for candidate in range(prev + 1, n + 1):
            count = math.comb(n - candidate, remaining)
            if rank0 >= count:
                rank0 -= count
            else:
                combo.append(candidate)
                prev = candidate
                break
    return tuple(combo)


T0_PRED8 = time.time()

# Rolling FD ima ACF signal. Prvo procenjujemo sledeci FD AR(1) logikom,
# zatim FD rezim koristi samo za amplitudu skoka, ne za smer.
rho_fd = float(acf_fd_roll[1]) if len(acf_fd_roll) > 1 else 0.0
fd_mean = float(fd_roll.mean())
fd_std = float(fd_roll.std(ddof=1))
last_fd = float(fd_roll[-1])
pred_fd = fd_mean + rho_fd * (last_fd - fd_mean)

local_window = fd_roll_window
local_y = np.asarray(lex_idx[-local_window:], dtype=float)
local_x = np.arange(len(local_y), dtype=float)
local_slope, local_intercept = np.polyfit(local_x, local_y, 1)
local_fit = local_intercept + local_slope * local_x
local_resid = local_y - local_fit
local_resid_std = float(local_resid.std(ddof=1))

recent_incr = np.diff(local_y)
recent_mean_incr = float(recent_incr.mean())
recent_std_incr = float(recent_incr.std(ddof=1))
last_lex = float(lex_idx[-1])
last_incr = float(np.diff(lex_idx)[-1])

fd_z = (pred_fd - fd_mean) / (fd_std + 1e-12)
amplitude_scale = float(np.clip(1.0 + 0.25 * fd_z, 0.50, 1.50))
direction = 1.0 if ((recent_mean_incr + last_incr) / 2.0) >= 0 else -1.0
base_magnitude = abs(0.5 * recent_mean_incr + 0.5 * last_incr)
pred_incr = direction * base_magnitude * amplitude_scale
pred_lex_float = last_lex + pred_incr
pred_lex = int(np.clip(round(pred_lex_float), 1, TOTAL_COMBOS))
pred_combo = lex_unrank_1based(pred_lex)

z_grid = [-1.28, -0.84, -0.43, 0.0, 0.43, 0.84, 1.28]
candidate_rows = []
seen_lex = set()
for z in z_grid:
    cand_lex = int(np.clip(round(pred_lex_float + z * recent_std_incr * amplitude_scale), 1, TOTAL_COMBOS))
    if cand_lex in seen_lex:
        continue
    seen_lex.add(cand_lex)
    candidate_rows.append((z, cand_lex, lex_unrank_1based(cand_lex)))

print()
print("PREDIKCIJA 8 — NEXT8 / 2c3b / rolling FD ACF")
print(f"  rho_FD lag-1           = {rho_fd:.8f}")
print(f"  FD mean                = {fd_mean:.8f}")
print(f"  FD std                 = {fd_std:.8f}")
print(f"  zadnji FD              = {last_fd:.8f}")
print(f"  pred. FD               = {pred_fd:.8f}")
print(f"  FD z                   = {fd_z:.6f}")
print(f"  amplitude scale        = {amplitude_scale:.6f}")
print(f"  lokalni slope          = {local_slope:,.2f}")
print(f"  recent mean dX         = {recent_mean_incr:,.2f}")
print(f"  recent std dX          = {recent_std_incr:,.2f}")
print(f"  zadnji dX              = {last_incr:,.2f}")
print(f"  pred. inkrement        = {pred_incr:,.2f}")
print(f"  pred. lex              = {pred_lex:,}")
print(f"  pred. kombinacija      = {pred_combo}")
print("  kandidati oko FD-amplitude prognoze:")
for z, cand_lex, combo in candidate_rows:
    print(f"    z={z:>5.2f}  lex={cand_lex:>10,}  combo={combo}")
print()

with open(TXT_PATH, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("PREDIKCIJA 8: NEXT8 / 2c3b / rolling FD ACF\n")
    f.write("=" * 60 + "\n\n")
    f.write("Model:\n")
    f.write("  Rolling FD ima ACF signal iznad shuffled reference.\n")
    f.write("  Sledeci FD se procenjuje AR(1): FD_next = mean(FD) + rho_FD*(FD_last-mean(FD)).\n")
    f.write("  FD se koristi kao rezim amplitude: visi FD -> veci skok, nizi FD -> manji skok.\n\n")
    f.write("Parametri:\n")
    f.write(f"  rho_FD lag-1           = {rho_fd:.8f}\n")
    f.write(f"  FD mean                = {fd_mean:.8f}\n")
    f.write(f"  FD std                 = {fd_std:.8f}\n")
    f.write(f"  zadnji FD              = {last_fd:.8f}\n")
    f.write(f"  pred. FD               = {pred_fd:.8f}\n")
    f.write(f"  FD z                   = {fd_z:.8f}\n")
    f.write(f"  amplitude scale        = {amplitude_scale:.8f}\n")
    f.write(f"  local window           = {local_window}\n")
    f.write(f"  lokalni slope          = {local_slope:,.8f}\n")
    f.write(f"  lokalni resid std      = {local_resid_std:,.8f}\n")
    f.write(f"  recent mean dX         = {recent_mean_incr:,.8f}\n")
    f.write(f"  recent std dX          = {recent_std_incr:,.8f}\n")
    f.write(f"  zadnji dX              = {last_incr:,.8f}\n")
    f.write(f"  zadnji lex             = {int(last_lex):,}\n")
    f.write(f"  pred. inkrement        = {pred_incr:,.8f}\n\n")
    f.write("Glavna prognoza:\n")
    f.write(f"  pred. lex float        = {pred_lex_float:,.8f}\n")
    f.write(f"  pred. lex              = {pred_lex:,}\n")
    f.write(f"  pred. kombinacija      = {pred_combo}\n\n")
    f.write("Kandidati oko FD-amplitude prognoze:\n")
    f.write(f"  {'z':>8}{'lex':>14}  kombinacija\n")
    for z, cand_lex, combo in candidate_rows:
        f.write(f"  {z:>8.2f}{cand_lex:>14,}  {combo}\n")
    f.write("\n")
    elapsed_pred8 = time.time() - T0_PRED8
    f.write(f"Vreme PREDIKCIJE 8: {timedelta(seconds=int(elapsed_pred8))} ({elapsed_pred8:.1f} s)\n")

print(f"TXT updated → {TXT_PATH}")
print(f"Vreme PREDIKCIJE 8: {timedelta(seconds=int(time.time()-T0_PRED8))} "
      f"({time.time()-T0_PRED8:.1f} s)")
print()


"""
ACF signal rolling FD: prvo procena sledeće lokalne hrapavosti, pa time podešavam širinu/skok inkrementa.

AR(1) procena sledećeg rolling FD preko rho_FD = ACF lag-1
FD koristi kao režim amplitude: veći FD → veći očekivani skok, manji FD → manji skok
smer uzima iz lokalnog prosečnog i zadnjeg inkrementa
generiše glavnu Loto kombinaciju + kandidate
upisuje u 3_KarlWeierstrass_NEXT8_2c3b.txt
"""



"""
3_KarlWeierstrass_NEXT8_2c3b — KORAK 1: formiranje krive f(t)
  CSV:                  /data/loto7_4624_k43.csv
  Ucitano izvlacenja:   4624
  C(39,7):              15,380,937


KORAK 2c3b: Aparat 2c Fraktalna dimenzija + Test 3b Autokorelacija (ACF)
  rolling FD max |ACF| lag 1..20: 0.5257
  95% band: +/-0.3520   znacajnih lagova: 2/20
  Ljung-Box aproks. h=10: Q=30.45  p=0.0007
  shuffled max|ACF|: mean=0.3119 std=0.0740 z=2.89 p=0.0140
  ⇒ rolling FD ima ACF signal iznad shuffled reference

PNG saved → /3_KarlWeierstrass_NEXT8_2c3b.png
TXT saved → /3_KarlWeierstrass_NEXT8_2c3b.txt
Vreme KORAKA 2c3b: 0:00:06 (6.7 s)
Ukupno vreme:      0:00:06 (6.8 s)

KRAJ 3_KarlWeierstrass_NEXT8_2c3b.


PREDIKCIJA 8 — NEXT8 / 2c3b / rolling FD ACF
  rho_FD lag-1           = 0.33475809
  FD mean                = 1.99750412
  FD std                 = 0.00279157
  zadnji FD              = 1.99932467
  pred. FD               = 1.99811357
  FD z                   = 0.218316
  amplitude scale        = 1.054579
  lokalni slope          = -739.26
  recent mean dX         = -6,859.23
  recent std dX          = 6,269,244.17
  zadnji dX              = -2,143,496.00
  pred. inkrement        = -1,133,859.79
  pred. lex              = 1
  pred. kombinacija      = (1, 2, 3, 4, 5, 6, 7)
  kandidati oko FD-amplitude prognoze:
    z=-1.28  lex=         1  combo=(1, 2, 3, 4, 5, 6, 7)
    z= 0.43  lex= 2,222,162  combo=(1, x, 14, y, 18, z, 26)
    z= 0.84  lex= 4,932,842  combo=(2, x, 19, y, 29, z, 37)
    z= 1.28  lex= 7,841,864  combo=(4, x, 13, y, 16, z,35)

TXT updated → /3_KarlWeierstrass_NEXT8_2c3b.txt
Vreme PREDIKCIJE 8: 0:00:00 (0.0 s)
"""
