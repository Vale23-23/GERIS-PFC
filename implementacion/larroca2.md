# Second review — GERIS-PFC FDCA implementation

**Repository:** `Vale23-23/GERIS-PFC`, branch `testing-fdca-real`, commit `db2f37b` (previous review: `5cd677f`)
**Reference:** ABI FDC ATBD v2.7, October 2020 (Schmidt, Hoffman, Prins & Lindstrom, UW-Madison SSEC/CIMSS)
**Attribution:** **[ATBD]** = stated in the reference document. **[COMMENT]** = my inference or recommendation.

---

## 0. Status of the previous findings

**Test suite: 18 passed, 0 failed.** All modules compile. `__pycache__` is gone from the tree.

| Previous | Status | Note |
|---|---|---|
| C1 categories 13/14 unreachable | **still open** — see R1 | fix changed the failure mode, not the outcome |
| C2 Dozier gets wrong background | **fixed** | `Tbc14` passed; `_background_radiances(Tb, …)` now used by *both* bisection and Newton, so the two stages solve the same system |
| C3 background cache | **fixed** | cache removed entirely, per-pixel computation |
| C4 all-`True` land mask | **fixed, with a side effect** — see R2 | `land_mask = region_mask & (eco_mask == 0)` |
| C5 TPW LUT | **partially addressed** — see R7 | `--tpw-lut-mode null` control implemented; provenance still undocumented |
| C6 `FailChar` 7 overwritten | **fixed** (mostly) — see R13 | `fc = int(fail_char_arr[i, j])` |
| C7 saturated shortcut | **fixed** | OR semantics, and diverted pixels now become candidates with `Tt = 0` |
| C8 glint flag on both branches | **fixed** | resolved explicitly in favour of Table 3.5, with the reasoning in the comment |
| H1 `Vis_Brightness_Value` conflation | **fixed** | separate `_histogram_select(vis_vals)` for `Histogram_Bin_Largest_Count` |
| H2 `FailChar` 8 set in §3.4.2.8 | **fixed** — see R8 | replaced by `smoke_corrected`, which is now dead |
| H3 emissivity asymmetry | **fixed** (mostly) — see R4 | background radiances now divided by emissivity |
| H4 0.38 → 0.07 | **documented** in `constants.py` |  |
| H5 bisection bracket check | **fixed** | `sign_hi` computed, non-bracketed and non-finite cases raise |
| H6 `abs()` on signed tests | **fixed, with a side effect** — see R3 |  |
| H7 extra along-scan condition | **fixed** |  |
| H8 pixel area | **fixed** | all four sides computed and opposite sides averaged |
| H9 FRP for codes 12/15 | **fixed** | `base_fire_code` captured before the temporal offset — correct, that ordering matters |
| H10 `n_passes > 10` dead | **documented** in `constants.py` |  |
| M1 night `cos(SZA)` in fog test | **fixed** |  |
| M2 `MIN_BT7` | **fixed** |  |
| M3 `std_dev_7_14_diff` | **fixed** | now follows the chosen approach |
| M4 audit trace logged corrected BT | **fixed** |  |
| M5 `state.py` / `temporal_filter.py` | **fixed** | code 12/32 added |
| M6 `ESUN_B02` | **fixed** | `kappa0` required, with an actionable error message |
| M8 / M9 / M10 | **documented** as deviations |  |
| L1 broken tests | **fixed** | `_make_fire_radiances` now builds and inverts with the same coefficients |
| L3 return annotations | **half fixed** — see R9 |  |

The `0.0 < elapsed` change in the temporal filter is also right — reprocessing the same timestamp no longer self-contaminates through `prev_fire_mask.npy`.

---

## 1. R1 — CRITICAL: categories 13 and 14 are still effectively unreachable

`part2.py::_high_med_thresholds`

The change was `t7_t14_diff = cand.bt7_bkg - cand.bt14_bkg` → `cand.bt7 - cand.bt14`. That removed the *algebraic* impossibility, but the test is still `Tb3.9 − Tb11.2 > add + off + (T3.9 − T11.2) + 2σ`, i.e. the **background** difference on the left and the **observed** difference inside the threshold. For a sub-pixel fire the observed difference is always the larger of the two — that is the fire signature — so the inequality runs the wrong way.

I sampled 3,000,000 physically reachable states (pixel excess at 3.9 µm in 0–60 K, at 11.2 µm in 0–15 K, background difference −2 to 8 K, σ 0–6 K, `n_passes` 0–10):

| reading | threshold-2 alone | with threshold-1 as well |
|---|---|---|
| previous code (`Db` in both places) | 0 / 3,000,000 | 0 |
| **current code** (`Db` vs `Dp` in threshold) | **2,853 / 3,000,000 (high)**, 15,837 (medium) | **8 (high)**, 484 (medium) |
| swapped (`Dp` vs `Db` in threshold) | 1,989,518 (high) | 1,969,811 |

The handful of current-code hits all require the 11.2 µm excess to exceed the 3.9 µm excess, which is backwards for a sub-pixel fire. A concrete marginal detection — background 300.0 / 296.0 K, pixel 312.0 / 297.0 K, σ = 1 K, one window expansion:

```
background diff Db = 4.0 K ; observed diff Dp = 15.0 K ; T3.9 - Tb3.9 = 12.0 K
previous code: thr2 = 12.0, LHS =  4.0  ->  falls through -> LOW_PROB (15)
current code : thr2 = 23.0, LHS =  4.0  ->  falls through -> LOW_PROB (15)
swapped      : thr2 = 12.0, LHS = 15.0  ->  HIGH_PROB (13)
```

**[ATBD]** §3.4.2.15 (p. 41) reads: *"T3.9 – Tb3.9 > first … threshold AND **Tb3.9 – Tb11.2** > second … threshold"*, with the second threshold built from *"5 (high) or 3 (medium) plus the 3.9 µm minus 11.2 µm temperature difference plus 2 times the standard deviation of the 3.9 µm minus 11.2 µm temperature difference within the background window."*

**[COMMENT]** I read the ATBD as having swapped the two difference terms. The evidence:

- Threshold 1 has the structure *observed excess > constant + window offset + 2 × background σ*. Threshold 2 is described as *"determined in a similar manner"*, so it should have the same structure — observed quantity on the left, background-derived floor on the right.
- In threshold 1 the standard deviation is explicitly qualified *"of the 3.9 µm (Channel 7) **background** temperature"*. In threshold 2 the standard deviation carries the qualifier *"within the background window"* but the bare difference term does not — consistent with the bare term being the **background** difference and the left side being the observed one.
- Only that arrangement produces a test that a real fire can pass.

So: put `T3.9 − T11.2` on the left and `Tb3.9 − Tb11.2` in the threshold, the reverse of the current assignment. And add a unit test that asserts both branches are reachable for a synthetic fire — this is the second time the same test has been silently unsatisfiable, and a reachability assertion is the only thing that catches it. `test_fire_mask_codes_valid` checks that emitted codes are *valid*, not that 13 and 14 are ever *emitted*.

---

## 2. R2 — HIGH (new): the country polygon is being used as the land mask

`fdca_adapter.py::build_surface_masks`, `load_fdca_input:1231-1238`

```python
region_mask = build_region_mask(lat, lon, region_name=region_name, base_path=base_path)
...
masks["land_mask"] = region_mask & (eco_mask_fixed == 0)
```

The `eco_mask == 0` half is exactly the C4 fix I recommended. The `region_mask &` half is new, and it conflates two different things.

**[ATBD]** §3.4.2.5 (p. 23): *"valid pixels are defined as **land pixels (as defined by the ancillary land-type data)**"*. A national border is not land-type data. Land on the Argentine and Brazilian sides of the border is perfectly valid background for a fire in Artigas, Rivera or Cerro Largo — which are among the most fire-prone departments in the country.

Out-of-region pixels still count toward `window_size` (correct, per the ATBD's out-of-bounds rule) but can never be valid, so border-adjacent pixels need more expansions to reach 20 % validity and are more likely to end at code 170. Uruguay is small relative to the maximum window:

```
country extent ~ 496 km x 544 km ; max window half-width = 110 km (55 px x 2 km)
fraction of area where a full 111x111 window fits inside the polygon:  33.1%
  11x11 (n_passes=0) : 92.4% unaffected
  31x31 (n_passes=2) : 78.2% unaffected
  61x61 (n_passes=5) : 59.1% unaffected
```

On a clear scene the effect is small; on a partly cloudy scene, where windows routinely expand, more than 40 % of the domain gets a degraded background — and the degradation is worst precisely at the borders.

**[COMMENT]** Separate the two roles:

- `region_mask` → output clipping only (which pixels get a reported code), which is what `part1.py:490` already does.
- `land_mask` → `eco_mask == 0` over a domain **padded by at least 55 pixels (110 km) beyond the ROI**, so background windows can draw on real land outside Uruguay.

That requires the ABI crop and the ancillary masks to be generated on the padded domain rather than the ROI. It is a change to the data-preparation step, not to the algorithm, and it is worth doing before Hito 1 because it changes background statistics over a large fraction of the country.

Two smaller points on the same code path:

- **Natural Earth 110m is too coarse for a 2 km grid.** `NATURAL_EARTH_URL` points at `110m_cultural`, a 1:110,000,000 dataset whose coastline and river boundaries are simplified by tens of kilometres. Used as a hard processing mask it will drop genuine land pixels along the Atlantic coast and the Río Uruguay. Use `10m_cultural/ne_10m_admin_0_countries.zip`.
- `build_region_mask` loops over every pixel in Python calling `geom.covers(point) or geom.contains(point) or geom.touches(point)`. `covers` already subsumes the other two. `shapely.vectorized.contains(geom, lon, lat)` (or `shapely.contains_xy` on Shapely ≥ 2.0) does the whole grid in one call.

---

## 3. R3 — HIGH: mask code 205 is now unreachable (side effect of my H6 recommendation)

`part1.py:568-571`

I recommended making both branches of the 2 K test signed, citing Table 3.11's wording. That was half right, and I should have checked the consequence. With both gates signed:

```python
if (bt7 > 273 or bt14_eff > 273) and diff_bt <= 2.0:      # gate 1 -> code 100, skip
    continue
if diff_bt < 2.0 and (bt7 <= 273 or bt14_eff <= 273):     # gate 2 -> code 201
    fire_mask = TOO_COLD; continue
```

Code 205 requires `T3.9 − T11.2 < −4`, which implies both `diff ≤ 2` and `diff < 2`. Gate 1 then requires *neither* channel above 273 K; gate 2 requires *both* above 273 K. Contradiction. Sampled over 5,000,000 random `(T3.9, T11.2)` pairs:

```
code 200 (T11.2 < 270)              : 2,211,516 / 5,000,000 reachable
code 205 (T3.9 - T11.2 < -4)        :         0 / 5,000,000 reachable
code 210 (diff > 20 and T3.9 < 285) :   513,126 / 5,000,000 reachable
```

**[COMMENT]** The reading that keeps every Table 3.11 code reachable is mixed:

- **Gate 1** (`|T3.9 − T11.2| ≤ 2` → code 100): **[ATBD]** §3.4.2.3 (p. 21) calls this *"a minimum threshold for fire activity"*. A pixel whose 3.9 µm is 10 K *colder* than its 11.2 µm has not failed a minimum fire threshold — it is a cloud, and it should proceed to the cloud tests. So the absolute value belongs here. This is what the code did before my review.
- **Gate 2** (`T3.9 − T11.2 < 2` → code 201): Table 3.11 names this *"negative difference threshold and below 273 K test"*, so signed, as it is now.

Keeping gate 2 signed while restoring the absolute value on gate 1 makes 200, 201, 205 and 210 all reachable. Whichever you choose, instrument it: print a histogram of `fire_mask` over a real scene and check that no Table 3.11 code you expect to see has a count of zero. That single diagnostic would have caught both this and R1.

---

## 4. R4 — MEDIUM: daytime FRP is biased low because the background still carries the solar component

`part1.py:1012-1021`

The emissivity fix brought pixel and background to the same level for TPW and emissivity. What remains is the solar reflectance term. `r7_diff` has `rad_solar` subtracted; `r7_bkg_corr` does not.

**[ATBD]** Eq. 3.4 (p. 34): `FRP_MIR = (A_pixel/a)·σ·(L_MIR − L_B,MIR)`. Since `L_MIR` has had the solar component removed and `L_B,MIR` has not, the difference is understated. The quantity that *is* the solar-free background 3.9 µm radiance is already computed and returned by `_solar_correction` as `emiss7 * rad7from14_bkg`.

Bias for a fire contributing 0.5 mW m⁻² sr⁻¹ (cm⁻¹)⁻¹ above background, over a range of background 3.9 µm solar offsets:

| `Tb3.9 − Tb11.2` | FRP bias |
|---|---|
| 1 K | −6.3 % |
| 2 K | −12.8 % |
| 3 K | −19.6 % |
| 5 K | −33.9 % |

**[ATBD]** §3.4.1 (p. 18) puts typical clear-sky differences at 2–5 K, so this is a systematic daytime low bias of roughly 13–34 % on the one product you have identified as the most trustworthy validation metric. Use `emiss7 * rad7from14_bkg` (equivalently `r7_bkg_corr − rad_solar`) as `L_B,MIR`.

---

## 5. R5 — MEDIUM (new): `OUT_OF_REGION = 255` duplicates an existing ATBD code

`constants.py:120`, `part1.py:491`

**[ATBD]** Table 3.11 (p. 43) already defines **code 0** as *"Non-processed region of input/output image"* — precisely what `OUT_OF_REGION` means — and Table 3.12 (p. 44) maps it to QA flag 3 (*"Pixel unusable due to unusable surface type, sunglint, or being off the disk"*). Emitting 255 makes the mask non-conformant with the documented code set, so any consumer validating against Table 3.11 (including your own `test_fire_mask_codes_valid` and `run_audit.py`'s `CODE_LABELS`) will not recognise it.

Use `FireMask.NON_PROCESSED` (0), which is already defined in the `FireMask` class and currently unused.

**[COMMENT]** Related gap, not an error: **Table 3.12 QA flags are not produced anywhere in the codebase.** For an output intended to reach firefighters or a national agency, the QA flag array is the ATBD's designated summary of the per-pixel mask and is probably worth adding before the deliverable — it is a pure function of `fire_mask`, so it costs a lookup table.

---

## 6. R6 — MEDIUM: the TPW LUT is still unverified

`fdca/data/tpw_lut.csv`

The `--tpw-lut-mode null` control in `run_audit.py::build_null_tpw_lut` is exactly the right instrument and is implemented correctly (transmittance rows set to 1, offset rows to 0, bin label rows preserved). But nothing in the repository records:

1. **the result of running it** — no comparison of real vs. null on a fixed scene;
2. **the provenance of the table** — which radiative transfer model, which spectral response functions, which atmospheric profiles;
3. **the units of each column**.

The two concerns from the first review are unchanged. The 11 µm offsets are 0.04–0.7 % of native Channel 14 radiance while the 3.9 µm offsets are 5–20 % of native Channel 7 radiance, which suggests the two were generated on different scales; and applying the TPW bin 5 row to a 300 K background gives `Tbc = 354 K`, a +54 K correction where +5 to +15 K is physical. Montevideo summer TPW lands in that bin.

**[COMMENT]** Run the null control on the same scene as `README_testing.md` §9.5 and record the detection-count delta by category. If the delta is large, the LUT is the dominant unknown in your Hito 1 error budget and regenerating it — with CRTM or RTTOV over South American profiles — becomes the highest-value remaining task, and a defensible scientific contribution in its own right.

---

## 7. LOW

| # | Location | Finding |
|---|---|---|
| R7 | `part1.py:834-844` | `smoke_corrected` is assigned in both branches and **never read**. The H2 fix correctly removed the wrong `FailChar.F8` write, and §3.4.2.9's F8 condition is implemented independently at line 923, so behaviour is right — but the variable is dead. Either delete it or use it to make the §3.4.2.8 → §3.4.2.9 hand-off explicit as the ATBD describes. |
| R8 | `part1.py:189` | `_tpw_lut_indices` still annotated `-> tuple[int, int]` while returning a single `int`. Commit `db2f37b` updated the docstring only. `_valid_background_mask`'s docstring was fixed but its signature has no annotation to correct. |
| R9 | `part1.py:729, 1035` | The `FireCandidate.bt_bkg_corr` field receives `bkg.temp7_bkg_mean` (uncorrected) on the quick path and `Tbc7` (Channel 7, not the ATBD's `Tbc`) on the main path. Nothing reads it today, which is the only reason it is harmless. Set it to `Tbc14` on both paths, or drop the field. |
| R10 | `dozier.py:41-46, 240-250` | `_solve_bisection`'s `_rad7_bkg`/`_rad14_bkg` are unused, and `compute_dozier`'s `rad7_bkg`/`rad14_bkg` now only reach `_solve_newton`'s validity guard. The comment says "retained for API compatibility", but the only caller is `part1.py`. Removing them makes it structurally impossible to reintroduce C2. |
| R11 | `run_fdca.py:510` | Uses `inp.region_mask` directly, while `run_part1.py:80` and `algorithm.py:160` use `getattr(inp, "region_mask", …)` with a default. Since `region_mask` is attached after construction rather than being a dataclass field, any `FDCAInput` not built by `load_fdca_input` will raise `AttributeError` here. Make it a proper `Optional` field on the dataclass. |
| R12 | `part1.py:900-907` | `FailChar.F7` survives to the post-correction tests but can still be overwritten by `fc = FailChar.F3` when a Channel 7 saturated pixel has `T11.2c < 285 K`. Harmless now that Part II keys FRP suppression on `base_fire_code` rather than the flag, and `is_saturated` is carried separately — but the audit flag is still lost for exactly the pixels you would most want to audit. |

---

## 8. Suggested order

1. **R1** — swap the two difference terms in threshold 2, then add the reachability test. Without this, categories 13 and 14 remain empty and the `LOW_PROB` pile-up in §9.5 will not move.
2. **R3** — restore the absolute value on gate 1 only, then print the full `fire_mask` code histogram on a real scene as a standing check.
3. **R2** — split `region_mask` from `land_mask` and regenerate the ancillary masks on a padded domain.
4. **R6** — run the null-LUT control and record the result; then decide whether to regenerate the table.
5. **R4** — one-line change to the FRP background term.
6. **R5**, then the LOW items.

## 9. Appendix material

Additions for **Appendix A (ATBD internal inconsistencies)** arising from this pass:

- §3.4.2.15 lists flag values *"3", "4", "6", or "8"* as eligible for the confidence upgrade, but flag 8 can never reach Part II: §3.4.2.10 overwrites it during Dozier under either resolution of the F9 contradiction. The `"8"` entry is dead under any reading.
- §3.4.2.10 describes the pixel area as a **rectangle** from averaged opposite sides; §3.4.2.11 describes it as **Heron's formula for a parallelogram**. The implementation follows the more detailed §3.4.2.10 description.
- The §3.4.2.3 gate ordering makes mask code 205 unreachable under a fully signed reading of the 2 K test (see R3).
- Table 3.11 defines code 0 for the non-processed region, which overlaps the ROI-clipping role the implementation gives to its own code (see R5).

Additions for **Appendix B (deliberate deviations)**: the F9 resolution in favour of Table 3.5 over §3.4.2.10; the rectangle-over-Heron choice for pixel area; the ROI clipping itself (no ATBD equivalent); and whichever reading of the 2 K test you settle on.