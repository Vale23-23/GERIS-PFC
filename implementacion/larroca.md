# Code review — GERIS-PFC FDCA implementation vs. ABI FDC ATBD v2.7

**Repository:** `Vale23-23/GERIS-PFC`, branch `testing-fdca-real`, commit `5cd677f`, directory `implementacion/`
**Reference:** *GOES-R Advanced Baseline Imager (ABI) Algorithm Theoretical Basis Document for Fire / Hot Spot Characterization*, Schmidt, Hoffman, Prins & Lindstrom, UW-Madison SSEC/CIMSS, **Version 2.7, October 2020** (the copy in the project folder). The header of `constants.py` correctly cites v2.7.

**Attribution convention** (as used elsewhere in this project):
**[ATBD]** = stated in the reference document. **[COMMENT]** = my inference, reading, or recommendation.

**Acronyms on first use:** FDCA = Fire Detection and Characterization Algorithm; ATBD = Algorithm Theoretical Basis Document; ABI = Advanced Baseline Imager; BT = brightness temperature; TPW = Total Precipitable Water; LUT = look-up table; FRP = Fire Radiative Power; SZA = solar zenith angle; LZA = local (satellite) zenith angle; DQF = Data Quality Flags; PUG = Product User Guide; WFABBA = Wildfire Automated Biomass Burning Algorithm; MIR = middle infrared.

---

## 0. Executive summary

The skeleton is sound and several genuinely hard pieces are right (see §5). The problems cluster in three places:

1. **Two ATBD branches are structurally unreachable** — fire categories 13 and 14 can never be assigned, and the `n_passes > 10` shortcut can never fire. This alone explains a large part of the "1782 of 2917 detections landed in `LOW_PROB`" symptom recorded in `README_testing.md` §9.5.
2. **The background estimate is contaminated three separate ways** — an all-`True` land mask, a caching bug that reuses a neighbour's (sometimes a different scan line's) window, and a TPW LUT that inflates the corrected background 11.2 µm BT by ~54 K at the TPW values typical of a Uruguayan summer.
3. **The Dozier inversion receives the wrong background temperature.** The solver itself is correct — I verified it recovers `(p, Tt)` exactly when fed a consistent background — but `part1.py` passes it the Channel 7 background temperature where **[ATBD]** Table 3.4 defines `Tbc` as the *corrected 11.2 µm* background, used for both channels. With a 4 K discrepancy (the magnitude the ATBD itself calls typical), retrieved fire fraction is off by 2–9× and fire temperature by 60–645 K.

Given the near-term milestone is *statistical* agreement with the operational NOAA FDC rather than pixel-for-pixel reproduction, I would fix **C1, C3, C4, C5** first — those four are the ones that move aggregate counts.

---

## 1. CRITICAL — changes detections or retrievals

### C1. Part II high/medium confidence thresholds are unsatisfiable; codes 13 and 14 are unreachable
`part2.py::_high_med_thresholds` / `_upgrade_confidence`

**[ATBD]** §3.4.2.15 (p. 41): the second confidence threshold is *"the larger value between 7 K (high) or 5 K (medium) and a scaled factor … the window size factor is then added to 5 (high) or 3 (medium) plus the 3.9 µm minus 11.2 µm temperature difference plus 2 times the standard deviation of the 3.9 µm minus 11.2 µm temperature difference within the background window."* The test it gates is `Tb3.9 – Tb11.2 > second threshold`.

The code interprets *"the 3.9 µm minus 11.2 µm temperature difference"* as the **background** difference:

```python
t7_t14_diff = cand.bt7_bkg - cand.bt14_bkg          # = D
scaled2 = add_c + bg_off + t7_t14_diff + 2.0 * diff_std
thresh2 = max(base, scaled2)
...
if bt7c - Tb7 > thr1_h and Tb7 - Tb14 > thr2_h and refl_test:   # D > thresh2
```

The same quantity `D` now appears on both sides. Writing it out:

- If `add + off + D + 2σ ≥ base`, the test becomes `D > add + off + D + 2σ`, i.e. `0 > add + off + 2σ` — false for all inputs, since `add ≥ 3`, `off ≥ 0`, `σ ≥ 0`.
- Otherwise `thresh2 = base` and the test is `D > 7` (or `> 5`), but that branch requires `D < 2 − off − 2σ ≤ 2`. Contradiction.

I ran 2,000,000 random draws over `D ∈ [−30, 30]`, `σ ∈ [0, 10]`, `n_passes ∈ [0, 10]`: **zero hits for both high and medium.** Every candidate reaching `_upgrade_confidence` therefore falls through to `return FireMask.LOW_PROB, 0`.

**[COMMENT]** The ATBD sentence is genuinely ambiguous, but the only reading that yields a satisfiable test is the **observed pixel** difference `T3.9 − T11.2`, not the background one. Change `t7_t14_diff` to `cand.bt7 - cand.bt14` and re-check. Whatever you decide, add an assertion or a unit test that both branches are reachable — this is exactly the class of error that a "does this condition ever evaluate true?" test catches.

### C2. Dozier receives the Channel 7 background temperature instead of `Tbc`; bisection and Newton solve different systems
`part1.py:866`, `dozier.py::compute_dozier`

**[ATBD]** Table 3.4 (p. 19): *"`Tbc` — Background temperature estimate corrected for atmospheric transmittance, emissivity, solar reflectivity, thin clouds/smoke. **This is equivalent to the corrected Tb11.2**, and is used for tests with both Channels 7 and 14."* §3.4.2.10 (p. 32) repeats it: *"Term C is the proportion of the total radiance due to the background non-fire portion of the pixel at Tb (here Tb is equivalent to Tbc)."*

`part1.py` computes both and passes the wrong one:

```python
Tbc7  = planck_temp_from_coeffs(r7_bkg_corr,  **coeffs7)     # Channel 7 background
Tbc14 = planck_temp_from_coeffs(r14_bkg_corr, **coeffs_long) # this is the ATBD's Tbc
...
doz = compute_dozier(r7_diff, r14_diff, r7_bkg_corr, r14_bkg_corr, Tbc7, ...)
```

There is a second, independent inconsistency inside `dozier.py`. `_solve_bisection` builds the background term from the two radiance arguments (`rad7_bkg`, `rad14_bkg`), while `_solve_newton` derives *both* from the single scalar `Tb`:

```python
Lb7  = planck_rad_from_coeffs(Tb, **coeffs7)
Lb14 = planck_rad_from_coeffs(Tb, **coeffs14)   # Channel 14 radiance at the Channel 7 background temperature
```

So the Newton stage refines a different system than the one bisection bracketed.

**Quantified.** I reconstructed the broken unit test (§4, L1) and drove `compute_dozier` with synthetic radiances built from a known `(p, Tt)`:

| `p` true | `Tt` true | consistent `Tb` → `p`, `Tt` | with `Tbc7 = Tbc14 + 4 K` → `p`, `Tt` |
|---|---|---|---|
| 0.005 | 800 K | 0.00500, 800.00 K | 0.00058, 1445.6 K |
| 0.010 | 800 K | 0.01000, 800.00 K | 0.00475, 949.2 K |
| 0.020 | 700 K | 0.02000, 700.00 K | 0.01283, 762.8 K |

The solver is correct; the inputs are not. A 4 K offset is not pathological — **[ATBD]** §3.4.1 (p. 18) says clear-sky shortwave/longwave window differences run *"on the order of 2-5 K due to reflected solar radiation, surface emissivity differences, and water vapor attenuation."* Since `fire_area = fire_frac × pixel_area`, retrieved fire size inherits the same error.

**[COMMENT]** Pass `Tbc14`, and make `_solve_bisection` derive its background radiances from that same `Tb` so the two stages are consistent by construction. This reinforces the note in your project memory that FRP is more trustworthy than the Dozier size/temperature — but here even FRP is affected, via C-issue H3.

### C3. Background window cache reuses a neighbour's window, and leaks across scan lines
`part1.py:641-652`

**[ATBD]** §3.4.2.5 (p. 23): *"Background statistics are updated along a given scan line; the calculation is skipped if the background statistics were calculated for the previous element."*

```python
if prev_j_bkg == j - 1 and prev_bkg is not None:
    bkg = prev_bkg
else:
    bkg = compute_background(i, j, ...)
    prev_j_bkg = j
    prev_bkg   = bkg
```

Two defects:

1. **`prev_j_bkg` is only updated in the `else` branch**, so along a dense row the pattern is compute, reuse, compute, reuse… Half of all pixels get a background window centred one element to their left. I simulated the exact logic: `j=0 compute, j=1 REUSE(from j=0), j=2 compute, j=3 REUSE(from j=2)…`
2. **Neither variable is reset at the start of each row `i`.** The comment claims *"the row boundary naturally breaks the sequence because j resets to 0"*, but that only holds if the previous row's last computed element was not `j=0`. Simulated counter-example: row 0 reaches the background step only at `j=500`; row 1's first qualifying pixel is `j=501`; it satisfies `prev_j_bkg == j-1` and **reuses a window centred on the previous scan line.** In a scene where clouds cause most pixels to exit early, this happens routinely.

**[COMMENT]** Reset `prev_j_bkg = None; prev_bkg = None` at the top of the `for i` loop, and update `prev_j_bkg = j` on the reuse path too. Beyond that, I'd read the ATBD sentence as describing an *optimisation for identical windows*, not an instruction to share statistics between distinct pixels — the safest interpretation for Hito 1 is to compute per-pixel and treat the cache purely as a speed-up keyed on whether the valid-pixel set actually changed.

### C4. `land_mask` is unconditionally `True`, so water is never excluded from background statistics
`fdca_adapter.py::build_surface_masks`

**[ATBD]** §3.4.2.5 (p. 23): *"For the purpose of background window calculations, valid pixels are defined as **land pixels** (as defined by the ancillary land-type data) and are subjected to rudimentary tests for warm pixels and clouds."*

```python
land_mask = np.ones((H, W), dtype=bool)
...
if region_name in ("rio_de_la_plata",):     # only this one region gets water
```

For `region_name="uruguay"` — the region the whole pipeline defaults to — every pixel is land. Meanwhile `data/eco_mask.npy` **does** carry the water information: I checked its contents and it contains 225 pixels coded 150 (invalid ecosystem type) and 852 coded 153 (inland water and other land/water mix) out of 67,872. Those 1,077 pixels are correctly excluded from *fire candidacy* in `part1.py`, but they still enter every background window that overlaps them.

**[COMMENT]** One line fixes it: `land_mask = (eco_mask_fixed == 0)`. Uruguay's fire-prone coastal and Laguna Merín margins are precisely where this bites — water pixels drag `Temp4_Bkg_Mean`/`Temp11_Bkg_Mean` cold and inflate the standard deviations, which loosens some tests and tightens others unpredictably. This is a plausible contributor to the over-detection in `README_testing.md` §9.5.

### C5. TPW LUT: the 11 µm column produces a physically implausible correction, and the two offset columns look mis-scaled
`fdca/data/tpw_lut.csv`, consumed by `part1.py::_apply_tpw_correction`

The LUT interface and indexing are **correct** — I checked `_tpw_lut_indices` against **[ATBD]** §3.4.2.8 (p. 29) row-by-row and column-by-column and it matches (`col = 7*(tpw_bin−1) + angle_bin − 1`, rows 3–6 → indices 2–5). The problem is the *contents*.

Working in ABI native radiance units (mW m⁻² sr⁻¹ (cm⁻¹)⁻¹), which is what the code feeds the LUT:

| Band | L at 300 K | LUT offset range | offset as % of L |
|---|---|---|---|
| B07 (3.9 µm) | 0.898 | 0.046 – 0.184 | 5 – 20 % |
| B14 (11.2 µm) | 118.7 | 0.044 – 0.817 | 0.04 – 0.7 % |

The 3.9 µm offsets are on a sensible scale. The 11.2 µm offsets are three orders of magnitude too small to matter, which makes the Channel 14 correction effectively a bare `1 / trans` division. That asymmetry suggests the two columns were generated on different radiance scales.

More seriously, the transmittances themselves: applying the TPW bin 5 row (TPW ≈ 50 mm, nadir angle bin) to a 300 K background gives

```
TPW bin 1: trans=0.949  ->  Tbc = 303.63 K  (+3.63 K)
TPW bin 5: trans=0.511  ->  Tbc = 354.12 K  (+54.12 K)
```

**[COMMENT]** A +54 K atmospheric correction on an 11 µm background is not physical; realistic surface-vs-TOA differences at 50 mm TPW are roughly +5 to +15 K. Montevideo summer TPW routinely sits in the 40–55 mm range, so **the season and region you care about lands in the worst bin.** A `Tbc` of ~354 K will fail the `T11.2c < 285 K` post-correction test in the wrong direction, break `T3.9c − Tbc` comparisons, and hand Dozier a background hotter than most fires.

Two things I'd want documented before trusting any Hito 1 number: (i) the provenance of `tpw_lut.csv` — which radiative transfer model, which spectral response functions, which atmospheric profiles — currently recorded nowhere in the repo; (ii) the units each column is expressed in. Your memory notes the option of building the interface with a null table first to assess sensitivity; given this result, I'd do exactly that (`trans = 1`, `offset = 0`) as a control run before anything else, so you can attribute detection changes to the LUT specifically.

### C6. `FailChar` 7 (saturated pixel) is silently overwritten
`part1.py:634` vs `part1.py:725`

```python
if sat_flag:
    fail_char_arr[i, j] = FailChar.F7      # line 634
...
fc = FailChar.NONE                          # line 709
...
fail_char_arr[i, j] = fc                    # line 725 — unconditional overwrite
```

Every saturated pixel that survives to line 725 loses its flag. Consequences: **[ATBD]** §3.4.2.12 (p. 36) says *"if the fire has Fire Mask codes 11, 12, or 15, it has no reported FRP"*, and the code implements that exclusion via `EXCLUDED_FRP_FLAGS = (F7, F9, F10)` — which now never matches F7, so **saturated fires receive an FRP they should not have**. Since `_dozier_rad_fire` never ran for them, that FRP is computed from a background that was never reconciled with a fire solution.

**[COMMENT]** Initialise `fc = int(fail_char_arr[i, j])` at line 709 instead of `FailChar.NONE`, or guard the write.

### C7. The saturated / max-passes shortcut uses AND instead of OR, and discards the pixel
`part1.py:691-694`

**[ATBD]** §3.4.2.7 (p. 28): *"The first test that identifies possible fire detections is applied to pixels that are either flagged as saturated or required more than 10 passes … There are two tests that if true will stop the algorithm from further processing the pixel and the algorithm skips ahead to the determination of pixel size (algorithm goes to the end of Section 3.4.2.11 …). **The tests must be false for the pixel to remain under consideration** as a potential fire pixel."*

```python
if sat_flag or n_pass > BKG_MAX_ITER:
    if ((bt7 - bt14_eff < std_7b14b) and (bt7 - bkg.temp7_bkg_mean < std_7b)):
        continue   # Not a fire
```

Two problems. First, *"The tests must be false"* (plural, both) means the pixel is diverted if **either** is true — the code requires both, so it diverts strictly fewer pixels. Second, `continue` **drops** the pixel, whereas the ATBD says it *"skips ahead to the determination of pixel size"* and lands at the end of §3.4.2.11 — confirmed by §3.4.2.11 (p. 36): *"This test is also applied to possible fire pixels that did not go through the 'last chance' tests, **having jumped here from Section 3.4.2.4**."*

**[COMMENT]** This matters because it is the only route by which saturated pixels become fires. **[ATBD]** §3.4.2.15 (p. 40) defines category 11 as *"the fire temperature solution reaches Part II of the algorithm with a value equal to 0 K"* — which requires the pixel to reach Part II at all. I read the ATBD as: divert to pixel-area determination and carry forward as a candidate. Flag this as one of the documented ATBD ambiguities and pick a reading explicitly.

### C8. Glint-flagged pixels get `FailChar` 9 on both branches, collapsing them all to category 12
`dozier.py:293-309`

```python
if Tt < MIN_FIRE_TEMP:
    result.fail_char = FailChar.F9 if is_potential_glint else FailChar.F6
    ...
if is_potential_glint:
    result.fail_char = FailChar.F9    # also when Tt >= 400 K
```

**[ATBD]** contradicts itself here, and the contradiction is worth recording rather than papering over:

- §3.4.2.10 (p. 35): *"If the flag code … had been set to '8' and the fire temperature solution is **greater than** 400 K, then the flag code … is set to a value of '9'."*
- Table 3.5 (p. 23): *"9 — If fire has FailChar=8 and the estimated sub-pixel fire temperature is **less than** 400 K."*

The code assigns F9 in *both* cases, which is neither reading. Downstream, `_assign_fire_category` maps F9 → `CLOUD_CONTAM` (12), and **[ATBD]** §3.4.2.12 denies FRP to code 12. Net effect: **every sunlit pixel with albedo ≥ 0.25 or albedo excess > 0.07 is demoted to "partially cloudy/smoke" and loses its FRP**, regardless of how hot the retrieved fire is. For daytime Uruguayan scenes over pasture and stubble, that is a lot of pixels.

The inline comment at line 307 (*"Actually if Tt > 400 and was glint→ set to 9 cleared to no-glint processed"*) reads like the author noticed the contradiction and did not resolve it. Resolve it explicitly.

---

## 2. HIGH — deviations from the documented algorithm

### H1. `Vis_Brightness_Value` conflates two distinct Table 3.6 quantities
`background.py:288-298`

**[ATBD]** §3.4.2.5 (p. 26): *"For daylit pixels, the Channel 2 approach is always based on a histogram. If the 'statistical' method is chosen, the background visible brightness is the brightness from Channel 2 corresponding to **Histogram_Bin_Largest_Count**. In the 'histogram' cases the Channel 2 background is **Vis_Diff_Histogram**. This value is known as Vis_Brightness_Value."*

Table 3.6 defines them differently: `Histogram_Bin_Largest_Count` is the mean background visible brightness *"determined using a histogram technique"* (i.e. a histogram of the visible brightness itself), whereas `Vis_Diff_Histogram` is the mean visible brightness *"determined from a Channel 7 minus Channel 14 histogram approach"* (i.e. the visible mean over the pixels selected by the thermal-difference histogram).

The code sets both fields to the same value:

```python
vis_mean_bkg = vis_hist_mean          # always the T7−T14-selected subset
...
histogram_bin_largest_count = float(vis_hist_mean),
vis_diff_histogram          = float(vis_hist_mean),
```

So the "statistical" branch silently uses the wrong quantity. This propagates through `_background_albedo` → `ABkg` → `albedo − ABkg`, which drives the semi-transparent smoke correction (§3.4.2.8), the `FailChar` 8/10 assignments (§3.4.2.9), and the Part II cloud-edge re-test (§3.4.2.14).

### H2. `FailChar` 8 is set inside §3.4.2.8, pre-empting the §3.4.2.9 glint test
`part1.py:762, 766` vs `part1.py:841-844`

**[ATBD]** §3.4.2.8 (p. 30) says a *flag indicating this condition* is set and used in §3.4.2.9; §3.4.2.9 (p. 31) then defines the actual condition for value 8: *"if Albedo is greater than or equal to 0.25 or if Albedo minus the background albedo is greater than 0.07, the pixel is given a flag value of '8'."*

The code writes `FailChar.F8` directly in the §3.4.2.8 correction block — including the **first** branch, which fires for `0.025 < ΔA < 0.07`. Because §3.4.2.9's assignment is guarded by `if fc == FailChar.NONE`, a pixel with `ΔA = 0.03` and albedo 0.10 keeps F8 even though it satisfies neither §3.4.2.9 condition. It then enters Dozier as `is_potential_glint=True` and, per C8, comes out as F9 → category 12.

**[COMMENT]** Use a separate boolean (`smoke_corrected`) for the §3.4.2.8 condition and let §3.4.2.9 own the F8 assignment.

### H3. Emissivity correction applied asymmetrically to pixel and background radiances
`part1.py:769-796, 930-939`

**[ATBD]** §3.4.2.8 (p. 30): *"The observed radiance, which has already been corrected for TPW attenuation, is adjusted by dividing by the emissivity to obtain a more accurate value for the actual emitting radiance of **the background and the observed pixel**."*

The code divides only the pixel radiances:

```python
r7_corr_em  = r7_corr  / em7        # pixel:      TPW + emissivity
r14_corr_em = r14_corr / em14
r7_bkg_corr  = _apply_tpw_correction(r7_bkg_raw,  offset7,  trans7)    # background: TPW only
r14_bkg_corr = _apply_tpw_correction(r14_bkg_raw, offset14, trans14)
```

Three downstream expressions then mix correction levels:

- **Channel 14 diffraction** (`r14_diff`): emissivity-corrected pixel minus `0.30 ×` a non-emissivity-corrected background. **[ATBD]** writes both terms unprimed (`radcorr,11` and `radcorr,background,11`), so the pixel term is the one that deviates.
- **`Tbc14`**, which Table 3.4 explicitly defines as corrected for *"atmospheric transmittance, **emissivity**, solar reflectivity, thin clouds/smoke"*, is computed from a TPW-only radiance.
- **FRP** (line 930-939) differences `r7_diff` (TPW + emissivity + solar + diffraction) against `r7_bkg_corr` (TPW only). Since **[ATBD]** Eq. 3.4 is `FRP = (A_pixel/a)·σ·(L_MIR − L_B,MIR)`, an inflated difference biases FRP high — systematically, and by the emissivity deficit (typically 2–8 % of the background radiance for vegetated surfaces).

### H4. Second smoke-correction threshold silently changed from 0.38 to 0.07
`part1.py:763`

**[ATBD]** §3.4.2.8 (p. 30): *"If the Albedo is great than 0.38 or the difference between the pixel and background albedos is greater than or equal to 0.38, the following corrections are applied: T3.9 = T3.9 + 0.7; T11.2 = T11.2 + 2.1."*

```python
elif alb_ij > CLOUD_ALBEDO_THRESH or alb_diff >= CLOUD_ADJ_ALBEDO_HIGH:   # 0.38 or 0.07
```

**[COMMENT]** The ATBD's 0.38 for an albedo *difference* is very likely a copy-paste from the absolute-albedo threshold on the same line, and 0.07 makes the branch continuous with the one above it. I think the change is right. But it is undocumented — nothing in the code or README says the ATBD value was overridden. This is the kind of thing that belongs in the deviations appendix.

### H5. Bisection has no bracket check; ATBD mask codes 182, 185, 186, 187 are never emitted
`dozier.py::_solve_bisection`

**[ATBD]** §3.4.2.10 (p. 35): *"One specific error that triggers the 'last chance' fire tests is if one of the intermediary fire solutions fails in the bisection technique … **because there is no sign difference between the intermediary solution and the upper and lower solution bounds**."* Table 3.11 (p. 43) reserves code **185** for *"Values used for bisection technique … are invalid"*, **186** for invalid Newton radiances, **187** for Newton processing errors, and **182** for errors converting adjusted temperatures to radiance.

`_solve_bisection` computes `sign_lo` but never `sign_hi`, never verifies the root is bracketed, and runs all 15 iterations regardless. Worse, `_dozier_rad_fire` at `p = 1e-6` can return a negative radiance, `planck_temp_from_coeffs` then returns `NaN`, and `np.sign(NaN − NaN)` is `NaN`; the comparison `sign_mid == sign_lo` evaluates `False`, so **NaN silently steers the bisection into the `p_hi = p_mid` branch**. The result is a plausible-looking but meaningless `p_mid` handed to Newton.

The code funnels every failure into `FailChar.F6` and `fire_mask = 180`. Table 3.5 defines F6 as *"Estimated sub-pixel fire temperature < 400 K"* — a different condition from "the solver failed". This makes the audit trail unusable for diagnosing convergence problems.

### H6. `abs()` added to two signed ATBD tests
`part1.py:558, 560, 828`

**[ATBD]** §3.4.2.3 (p. 21) describes the 2 K test without absolute value, and Table 3.11 names code 201 *"3.9 µm minus 11.2 µm **negative difference** threshold and below 273 K test"* — the word "negative" indicates a signed comparison. The code uses `abs(diff_bt)`, which routes strongly negative differences (cold cloud tops, `T3.9 − T11.2 ≈ −10 K`) past the 201 gate and into the cloud tests instead. Same issue at line 828 for `T11.2c − Tbc < 0.25 K`, where §3.4.2.9 is unambiguously signed.

### H7. Undocumented extra condition inside the along-scan radiance test
`part1.py:163-166`

**[ATBD]** §3.4.2.6 (p. 27): *"If the radiance differences between the pixels ±2 elements away is less than the previously defined Std. Dev. (Reflb) test value **and** the 3.9 µm brightness temperature … is less than … T3.9ReflThreshold, then the along scan-line radiance test is false."* Two conditions.

```python
if refl_diff_m2 < std_reflb and refl_diff_p2 < std_reflb:
    if abs(r_m2) < std_reflb and abs(r_p2) < std_reflb:      # <- not in the ATBD
        if bt7_ij < bt7_refl_thr:
            return False
```

The middle test additionally requires each neighbour's own `Refl` magnitude to be small. This makes the function return `False` less often, so `pass_along` is `True` more often, so the `FailChar` 1 and 2 rejections (which OR against `pass_along`) fire more often — **fewer detections**. Also note the `abs()` on `refl[i,j] − r_m2`: **[ATBD]** §3.4.2.13 (p. 38) lists the transported quantity as *"difference between the value of rdd for the pixel being evaluated (location i) and the pixel at location i−2"* and describes the test as checking whether the pixel value is *"significantly greater than"* the neighbours — signed.

### H8. Pixel area never averages opposite sides
`dozier.py::compute_pixel_area`

**[ATBD]** §3.4.2.10 (p. 35-36): *"the area of the pixel is calculated by finding the lengths of the sides of the pixel using the great circle equation and treating it as a rectangle by **finding the average length of the vertical and horizontal sides** … which are then averaged between the top and bottom and the left and right to create a rectangle that approximates the 4x4 box."*

The code computes one vertical side, one horizontal side, and the diagonal, then applies Heron's formula. The averaging of top/bottom and left/right is missing. **[COMMENT]** For ABI fixed-grid pixels over Uruguay (LZA ≈ 45–55° from GOES-19 at 75.2° W) the two vertical sides differ measurably; the resulting area error propagates directly into FRP, which is linear in `A_pixel`.

### H9. FRP is not zeroed for final categories 12 and 15
`part2.py:333`

**[ATBD]** §3.4.2.12 (p. 36): *"if the fire has Fire Mask codes **11, 12, or 15**, it has no reported FRP … For fire pixels with no FRP calculated, FRP is set equal to −9."*

Part I cannot know the final category — codes 12 and 15 are decided in Part II — yet Part II only nulls FRP for `n_passes > BKG_MAX_ITER` (which, per H10, never happens). So low-probability and cloud-contaminated fires carry FRP values the ATBD says should be −9.

### H10. `n_passes` can never exceed 10, silently disabling three ATBD branches
`background.py:216`, `constants.py`

**[ATBD]** §3.4.2.5 (p. 23): *"the window expands as a square with each iteration including an additional 5 lines and 5 elements in each direction for a maximum of 10 iterations (for a maximum of 111 x 111 pixels)."*

The two halves of that sentence are inconsistent: 10 iterations from an 11×11 start reaches 101×101, not 111×111. The code chose "initial window + 10 expansions", which reproduces the stated 111×111 maximum — a defensible reading, and I'd keep it. But the consequence is that `n_passes ∈ [0, 10]` always, so:

- `part1.py:691` — `n_pass > BKG_MAX_ITER` is unreachable (only `sat_flag` can trigger the §3.4.2.7 shortcut).
- `part1.py:930` — the `n_pass <= BKG_MAX_ITER` FRP gate is always true.
- `part2.py:333` — `cand.n_passes > BKG_MAX_ITER` is unreachable, so `FRP_NPASSES_EXCEEDED` is dead.

Part II's `cand.n_passes >= BKG_MAX_ITER` (condition 3 of §3.4.2.14) *is* reachable, so the boundary is genuinely load-bearing. **[COMMENT]** Document the choice, and record the 101-vs-111 contradiction in the ATBD-inconsistencies appendix.

---

## 3. MEDIUM

| # | Location | Finding |
|---|---|---|
| M1 | `part2.py::_reassign_fog_edge` | `sza_cos * 20` is applied unconditionally. At night `cos(SZA) < 0`, so the term is negative. Every other day/night threshold in the codebase zeroes the solar term at night; this one does not. |
| M2 | `part1.py:625` | Night along-scan test uses `MIN_BT` (200 K) where **[ATBD]** §3.4.2.4 (p. 22) specifies `T3.9 >= 150 K`. `MIN_BT7 = 150.0` is defined in `constants.py` and never used. Practically inert because the 200 K gate at line 514 already fired — which is itself an ATBD internal inconsistency worth logging. |
| M3 | `background.py:336` | `std_dev_7_14_diff` is always the statistical standard deviation, even when the histogram approach wins. It feeds `std_7b14b`, a primary threshold. **[ATBD]** §3.4.2.5 says the histogram approach *is* computed on the difference; which one to use downstream is unstated. Pick one and say so. |
| M4 | `part2.py:298` | The audit trace logs `cand.bt7_corr - cand.bt7_bkg` while `_upgrade_confidence` actually decides on `cand.bt7` (observed). **[ATBD]** Table 3.4 distinguishes `T3.9` from `T3.9c`; the decision uses the right one, the diagnostic does not. This will mislead you during Hito 1 validation. |
| M5 | `state.py` vs `temporal_filter.py` | `FIRE_CODES_FOR_STATE_UPDATE` omits 12/32; `temporal_filter.FIRE_CODES` includes them. The two modules disagree about whether a cloud-contaminated fire seeds the previous-fire mask. **[ATBD]** Table 3.10 says the mask records *"when a fire was last detected"* and 12 is a fire category. |
| M6 | `fdca_adapter.py:75` | `ESUN_B02 = 1622.088 # CHEQUEAR ESTE VALOR CUAL ES` — an unresolved TODO in the radiometric chain, and no Earth–Sun distance correction (±3.4 % annually). **[COMMENT]** ABI L1b reflective-band files carry `kappa0`, which converts radiance to reflectance factor directly and already includes the distance term: `refl = kappa0 * Rad`. Use it and delete both the constant and the TODO. Everything downstream of albedo (thresholds 0.38, 0.25, 0.15, 0.07) depends on this. |
| M7 | `fdca_adapter.py::load_data_quality` | DQF are loaded and deliberately unused. **[ATBD]** §3.4.2.1 (p. 19): *"Due to the remapping, ABI pixels containing saturated ABI samples must be flagged for the algorithm to perform to user expectations."* The deferral is documented, which is good practice — but it is a known gap against §3.4.2.10's assumption that *"sub-pixel detector saturations are flagged and available."* |
| M8 | `part1.py:238` | `rad_solar = max(0.0, ...)` clamps the solar component to non-negative. Not in the ATBD. Defensible, undocumented. |
| M9 | `part1.py:235` | The solar reflectivity correction is gated on `0 <= sza <= 85`. **[ATBD]** §3.4.2.8 applies it unconditionally (*"Any remaining difference between the background 3.9 µm and 11.2 µm brightness temperatures is assumed to be due to solar reflectivity"*). Physically the gate is sensible; it is still an inference. |
| M10 | `background.py:137` | The centre pixel is excluded from its own background window. Not stated in the ATBD; consistent with WFABBA heritage practice. Reasonable, undocumented. |
| M11 | `part1.py:211` | `_apply_tpw_correction` subtracts the offset once, whereas **[ATBD]** Eq. p. 29 literally reads `(rad − ext·rad_offset) / trans`. The docstring explains the reasoning (the LUT column *is* `ext`), and I agree with the resolution — this one **is** documented, and is the model for how the others should be handled. |

---

## 4. LOW — hygiene and maintainability

**L1 — the test suite is broken, including the only Dozier correctness test.** `python -m pytest implementacion/tests/ -q` gives **6 failed, 12 passed**. Five integration tests die on `TypeError: FDCAInput.__init__() missing 1 required positional argument: 'rad15'`, and `test_dozier_recovers_known_solution` dies on `compute_dozier() missing 2 required positional arguments: 'coeffs7' and 'coeffs14'`. Both are signature drift that was never propagated to the tests. That last one is the single most valuable test in the repository — it is what would have caught C2. `README_testing.md` §10 also points at `implementacion/fdca/tests/`, a path that does not exist (the tests are in `implementacion/tests/`).

**L2** — `implementacion/__pycache__/*.pyc` is committed, including stale `fdca.cpython-314.pyc` and `part1.cpython-314.pyc` for modules that have since moved. Add to `.gitignore` and `git rm --cached`.

**L3** — Wrong return annotations: `_tpw_lut_indices` is annotated `-> tuple[int, int]` and its docstring promises `(row_offset, col_index)`, but it returns a single `int`. `_valid_background_mask` is annotated as a 3-tuple and returns a 2-tuple.

**L4** — `planck.temp_to_rad_in_band` computes `rad_src` and discards it; the function reduces to `planck_rad(dst_band, T)`. Its docstring's claim that *"brightness temperature is band-independent for a blackbody"* muddles the concept it is trying to explain.

**L5** — `LAMBDA[13] = 10.35e-6` where the ATBD says 10.3 µm. `LAMBDA[7] = 3.90e-6` is the nominal, not the effective, central wavelength (GOES-19 ABI band 7 sits near 2569.6 cm⁻¹ ≈ 3.892 µm); it is used in `native_wavenumber_radiance_to_per_meter`, so it enters FRP as a ~0.9 % systematic. The band's actual central wavenumber is in the L1b file.

**L6** — Mixed-language comments and shouty leftover TODOs (`# HAYQUE VER DONDE SE USA ESTO`, `#CHEQUEAR ESTE VALOR CUAL ES`, `# QUEDA PENDIENTE REVISAR SU USO`). Pick one language for code comments and move open questions to issues.

**L7** — `part1.py:884`: `glint_low_temp` is always `False`. It requires `fc == FailChar.F8` *and* `doz.valid` *and* `fire_temp < 400`, but `doz.valid` is only set when `Tt >= 400`, and lines 871-873 have already replaced `F8` with `F9`/`F6` by then. The pixel still reaches the last-chance test via `not doz.valid`, so behaviour is unaffected — but the variable name suggests the ATBD condition is implemented when it is not.

**L8** — `part1.py:904-905` recomputes `compute_pixel_area` with identical arguments. The ATBD's recomputation exists because jumped pixels arrive with the sentinel `−9`; here area is already computed unconditionally at line 881, so the branch is a no-op.

**L9** — `check.py`, `inspect_bkg.py`, `test_script.py` are scratch scripts with hardcoded paths sitting alongside the package.

---

## 5. What is right — worth saying explicitly

Several of these are the parts that are easy to get wrong, and the students got them right:

- **Planck inversion uses the official ABI L1b coefficients** (`fk1`, `fk2`, `bc1`, `bc2` per band per file, ABI L1b PUG §4.2.4) rather than a monochromatic Planck function. This is the correct band-integrated treatment of **[ATBD]** Table 3.8's *"radiance calculated by integrating the product of the Planck function and the response function."* The derivative `planck_deriv_T_from_coeffs` is analytically correct in the same parameterisation.
- **`native_wavenumber_radiance_to_per_meter` is dimensionally correct.** `L_λ = L_ν̄ · 1e-5 / λ²` follows exactly from `ν̄ = 1/(100λ)` plus mW→W. I verified it.
- **`FRP_MIR_A = 3.0e-3` is the correct unit conversion** of the ATBD's `a = 3.0 × 10⁻⁹ W m⁻² sr⁻¹ µm⁻¹ K⁻⁴` to per-metre radiance. The commit history shows this was found and fixed; it was worth fixing.
- **The hybrid longwave band construction matches §3.4.2.2 exactly** — both radiance differences taken in Channel 7 space, smallest absolute value selected per pixel.
- **The band-12/13 typo in §3.4.2.2 was read correctly.** The ATBD says *"If the radiances for bands 7, **12**, or 14 are found to be negative"*; the code handles 7/13/14, and the `use_ch13 & (rad13 < 0)` guard is a genuinely careful touch.
- **The 80° block-out uses LZA, not SZA.** §3.4.2.3 says "SZA", but Table 3.11 code 50 says *"Local zenith angle block-out zone"*. The code follows Table 3.11. Correct.
- **All four contextual threshold scalings in `_contextual_thresholds` match §3.4.2.6 precisely** — the ×3.0/cap 4.0, the ×3.75 + `min(5, n/3)` clamped [4, 10], the ×3.0 clamped [0.25, 1.0], and the ×2.5 + `0.5·min(5, n/3)` clamped [2.5, 10]. I checked each against the text.
- **B02 is reduced by 4×4 block averaging**, which is §3.4.2.1's *operational* choice (the development system uses the upper-left corner and *"will produce different results"*). Since the benchmark is the operational NOAA product, this is the right pick — say so in the deviations appendix.
- **Out-of-bounds pixels count toward the background window denominator but never as valid**, exactly as §3.4.2.5 specifies. Easy to get wrong.
- **Part II §3.4.2.14 conditions 1–3 and the temporal filter** (12 h = 43200 s, ±1 pixel, +20 code offset) match the text.
- **The `Stage` instrumentation in `part1.py`** is good engineering and directly serves the Hito 1 goal — being able to say *which gate* killed the pixels NOAA detected is exactly the right diagnostic.
- **`detection_policy="conservative"` is correctly quarantined** from the ATBD path with an explicit docstring saying it is an empirical calibration. This is the right pattern; apply it to H4, M8, M9, M10.

---

## 6. Suggested order of work

1. **C1** (unreachable categories 13/14) — one line, and it directly explains the `LOW_PROB` pile-up in `README_testing.md` §9.5.
2. **C3 + C4** (background contamination) — a few lines each, and they change aggregate detection counts more than anything else here.
3. **L1** (repair the test suite) — before touching anything else, so the remaining fixes are protected. Then add the reachability assertions.
4. **C5** (TPW LUT) — run the null-table control (`trans = 1`, `offset = 0`) first to quantify sensitivity, then document provenance and units.
5. **C2 + C6 + C7 + C8** (Dozier inputs, flag lifecycle) — these govern fire characterisation and category assignment.
6. **H1–H10**, then M and L.

## 7. For the deliverable

Two appendices are worth adding to the students' report, since several findings above are ATBD problems rather than implementation problems:

**Appendix A — ATBD internal inconsistencies encountered.** The 10-iterations-vs-111×111 arithmetic (§3.4.2.5); `FailChar` 9 defined with opposite inequalities in §3.4.2.10 and Table 3.5; `FailChar` 4 defined against the *unadjusted* Channel 14 BT in Table 3.5 but against `Tbc` in §3.4.2.9; the 0.38 albedo-difference threshold in §3.4.2.8; `T3.9 >= 150 K` in §3.4.2.4 being unreachable behind the 200 K gate in §3.4.2.3; the band-12/13 typo; SZA-vs-LZA in §3.4.2.3; FRP in kW (§3.4.2.13) vs MW (Table 3.10); FRP set to −9 (§3.4.2.12) vs −99 (§3.4.2.15) for the same `n_passes > 10` condition; and the §3.4.2.14 cross-reference pointing at §3.4.2.4 for a test that is actually defined in §3.4.2.6.

**Appendix B — deliberate deviations.** Every place where the code departs from the literal text, with the reasoning: the single-subtraction TPW correction (already documented), the Newton tolerance change from 1e-20 absolute to 1e-6 relative (documented in `constants.py`, should be promoted), the 0.38→0.07 albedo-difference threshold, the `max(0, ·)` solar clamp, the daytime gate on the solar correction, the centre-pixel exclusion from the background window, the operational B02 averaging choice, and the FPT proxy via `fpt_threshold_exceeded_count`.
