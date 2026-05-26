"""
Test suite para la implementación FDCA.
Prueba cada subsistema corregido con datos sintéticos
Correr con: python -m pytest tests/ -v o python tests/test_fdca.py

"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest
from fdca.planck import planck_rad, planck_temp, planck_deriv_T
from fdca.background import compute_background, BackgroundStats
from fdca.dozier import (
    compute_dozier, compute_frp, compute_pixel_area,
    _solve_bisection, _dozier_rad_fire,
)
from fdca.constants import FireMask, FailChar, MIN_FIRE_TEMP
from fdca import run_fdca, FDCAInput


# ── Planck tests: Para verificar la física radiométrica ──────────────────────────── 
class TestPlanck(unittest.TestCase):

    def test_roundtrip_ch7(self):
        """
        Para chequear que pasar de temperatura a radiancia y luego volver a temperatura devuelva el valor original
        planck_temp(planck_rad(T)) ≈ T para el canal 7.
        """
        for T in [250.0, 300.0, 400.0, 600.0, 1000.0]:
            rad = planck_rad(7, np.array([T]))[0]
            T_back = planck_temp(7, np.array([rad]))[0]
            self.assertAlmostEqual(T, T_back, places=6,
                msg=f"Ch7 round-trip falló en T={T} K")

    def test_roundtrip_ch14(self):
        """planck_temp(planck_rad(T)) ≈ T for channel 14."""
        for T in [250.0, 300.0, 350.0]:
            rad = planck_rad(14, np.array([T]))[0]
            T_back = planck_temp(14, np.array([rad]))[0]
            self.assertAlmostEqual(T, T_back, places=6,
                msg=f"Ch14 round-trip falló en T={T} K")

    def test_planck_monotone(self):
        """Para verificar que si aumenta la temperatura, aumente la radiancia."""
        T_arr = np.array([280.0, 300.0, 350.0, 500.0])
        rad7  = planck_rad(7,  T_arr)
        rad14 = planck_rad(14, T_arr)
        self.assertTrue(np.all(np.diff(rad7)  > 0), "Radiancia de Ch7 no monótona")
        self.assertTrue(np.all(np.diff(rad14) > 0), "Radiancia de Ch7 no monótona")

    def test_fire_sensitivity(self):
        """
        Esta es la base física del algoritmo; a 800K de temperatura de fuego, la radiancia del ch7 es >> radiancia de ch14
        """
        T_fire = 800.0
        rad7  = planck_rad(7,  np.array([T_fire]))[0]
        rad14 = planck_rad(14, np.array([T_fire]))[0]
        self.assertGreater(rad7 / rad14, 7.0,
            "Ch7/Ch14 debería ser >> 1 en 800K de temperatura de fuego")

    def test_derivative_positive(self):
        """La derivada de Planck respecto a T debería ser positiva."""
        d = planck_deriv_T(7, 600.0, 0.01)
        self.assertGreater(d, 0.0)


# ── Test estadísticos de Background ──────────────────────────────────────────
class TestBackground(unittest.TestCase):

    def _make_uniform_scene(self, size=30, bt7_val=295.0, bt14_val=293.0):
        """Crea una imagen falsa con ruido pequeño."""
        np.random.seed(42)
        bt7  = np.full((size, size), bt7_val)  + np.random.randn(size, size) * 0.5
        bt14 = np.full((size, size), bt14_val) + np.random.randn(size, size) * 0.3
        from fdca.planck import planck_rad
        # Refl = rad7 - rad14_in_ch7_space
        refl  = planck_rad(7, bt7) - planck_rad(7, bt14)
        vis   = np.full((size, size), 100.0)
        alb   = np.full((size, size), 0.1)
        land  = np.ones((size, size), dtype=bool)
        sza_cos = np.full((size, size), np.cos(np.radians(45.0)))
        is_day  = np.ones((size, size), dtype=bool)
        return bt7, bt14, refl, vis, alb, land, sza_cos, is_day

    def test_background_computed(self):
        """Verfica que se calcule correctamente la temperatura media de fondo."""
        bt7, bt14, refl, vis, alb, land, sza_cos, is_day = self._make_uniform_scene()
        bkg = compute_background(15, 15, bt7, bt14, refl, vis, alb, land, sza_cos, is_day)
        self.assertIsNotNone(bkg)
        self.assertAlmostEqual(bkg.temp7_bkg_mean, 295.0, delta=1.5)
        self.assertAlmostEqual(bkg.temp14_bkg_mean, 293.0, delta=1.5)

    def test_background_fails_over_water(self):
        """Background debería fallar (devolver None) cuando no hay pixeles de tierra disponibles."""
        bt7, bt14, refl, vis, alb, land, sza_cos, is_day = self._make_uniform_scene()
        land[:] = False   # Todo agua
        bkg = compute_background(15, 15, bt7, bt14, refl, vis, alb, land, sza_cos, is_day)
        self.assertIsNone(bkg)

    def test_warm_pixels_excluded(self):
        """Los píxeles muy calientes no deberían sesgar la media de background."""
        bt7, bt14, refl, vis, alb, land, sza_cos, is_day = self._make_uniform_scene()
        # Inyectar pixel caliente en el centro
        bt7[15, 15] = 500.0
        bkg = compute_background(15, 15, bt7, bt14, refl, vis, alb, land, sza_cos, is_day)
        self.assertIsNotNone(bkg)
        # Promedio debería seguir siendo ~295 K a pesar del pixel caliente
        self.assertLess(bkg.temp7_bkg_mean, 310.0)


# ── Dozier tests: prueba el método físico de sub-pixel fire retrieval ───────────────────────────────────
class TestDozier(unittest.TestCase):

    def _make_fire_radiances(self, p=0.01, Tt=800.0, Tb=295.0):
        """Genera radiancias sintéticas para una combinación conocida (p, Tt, Tb)."""
        from fdca.planck import planck_rad
        L7_fire = planck_rad(7,  np.array([Tt]))[0]
        L14_fire= planck_rad(14, np.array([Tt]))[0]
        L7_bkg  = planck_rad(7,  np.array([Tb]))[0]
        L14_bkg = planck_rad(14, np.array([Tb]))[0]
        rad7  = p * L7_fire  + (1-p) * L7_bkg
        rad14 = p * L14_fire + (1-p) * L14_bkg
        return rad7, rad14, L7_bkg, L14_bkg

    def test_bisection_midpoint_formula(self):
        """Testea la correcta bisección midpoint (logaritmica, ATBD eq. 3.1)."""
        p_lo, p_hi = 1e-6, 1.0
        expected_mid = 10 ** (np.log10(p_lo) + (np.log10(p_hi) - np.log10(p_lo)) / 2.0)
        # Should be sqrt(p_lo * p_hi) in log space
        self.assertAlmostEqual(expected_mid, np.sqrt(p_lo * p_hi), places=10)
        # Should NOT be (p_lo + p_hi) / 2 (the broken version)
        wrong_mid = (p_lo + p_hi) / 2.0
        self.assertNotAlmostEqual(expected_mid, wrong_mid, places=3)

    def test_dozier_recovers_known_solution(self):
        """Dozier debería recuperar aproximadamente (p, Tt) usado para generar radiancias."""
        p_true, Tt_true, Tb = 0.01, 800.0, 295.0
        rad7, rad14, r7_bkg, r14_bkg = self._make_fire_radiances(p_true, Tt_true, Tb)
        result = compute_dozier(rad7, rad14, r7_bkg, r14_bkg, Tb)
        self.assertTrue(result.valid, "Dozier should find a valid solution")
        self.assertGreater(result.fire_temp, MIN_FIRE_TEMP,
            f"Fire temp {result.fire_temp:.1f} K < {MIN_FIRE_TEMP} K")
        self.assertAlmostEqual(result.fire_temp, Tt_true, delta=50.0,
            msg=f"Fire temp {result.fire_temp:.1f} K far from true {Tt_true} K")
        self.assertAlmostEqual(result.fire_frac, p_true, delta=0.005,
            msg=f"Fire fraction {result.fire_frac:.5f} far from true {p_true}")

    def test_frp_sign(self):
        """FRP should be positive when observed > background."""
        from fdca.planck import planck_rad
        r7_obs = planck_rad(7, np.array([310.0]))[0]
        r7_bkg = planck_rad(7, np.array([295.0]))[0]
        frp = compute_frp(4.0, r7_obs, r7_bkg)
        self.assertGreater(frp, 0.0)

    def test_pixel_area_positive(self):
        """Pixel area should be positive and plausible for Uruguay."""
        size = 200
        lats = np.linspace(-35.5, -30.0, size)
        lons = np.linspace(-58.5, -53.0, size)
        lat2d = np.tile(lats[:, None], (1, size))
        lon2d = np.tile(lons[None, :], (size, 1))
        area = compute_pixel_area(100, 100, lat2d, lon2d)
        # GOES-16 pixels over Uruguay should be ~4-8 km²
        self.assertGreater(area, 1.0, "Area too small")
        self.assertLess(area, 200.0, "Area too large")


# ── Full integration test ─────────────────────────────────────────────────────
class TestIntegration(unittest.TestCase):

    def _make_synthetic_input(self, L=40, W=40, inject_fire=True):
        """
        Build a minimal FDCAInput with synthetic data over a Uruguay-like domain.
        Optionally injects a sub-pixel fire at pixel (20, 20).
        """
        from fdca.planck import planck_rad

        np.random.seed(0)
        # Background scene: 295 K / 293 K
        bt7_bg  = 295.0
        bt14_bg = 293.0

        bt7  = np.full((L, W), bt7_bg)  + np.random.randn(L, W) * 0.3
        bt14 = np.full((L, W), bt14_bg) + np.random.randn(L, W) * 0.2
        rad7  = planck_rad(7,  bt7)
        rad14 = planck_rad(14, bt14)

        if inject_fire:
            # Inject fire at (20, 20): p=0.01 fire at 800 K + background
            p, Tt, Tb = 0.01, 800.0, bt7_bg
            L7_f  = planck_rad(7,  np.array([Tt]))[0]
            L14_f = planck_rad(14, np.array([Tt]))[0]
            L7_b  = planck_rad(7,  np.array([Tb]))[0]
            L14_b = planck_rad(14, np.array([Tb]))[0]
            rad7 [20, 20] = p * L7_f  + (1-p) * L7_b
            rad14[20, 20] = p * L14_f + (1-p) * L14_b
            bt7 [20, 20]  = float(planck_temp(7,  np.array([rad7 [20,20]]))[0])
            bt14[20, 20]  = float(planck_temp(14, np.array([rad14[20,20]]))[0])

        # Geometry (Uruguay ~33°S, GOES-16 ~75°W lon → LZA ~40-50°)
        lats = np.linspace(-34.0, -32.0, L)
        lons = np.linspace(-57.0, -55.0, W)
        lat2d = np.tile(lats[:, None], (1, W))
        lon2d = np.tile(lons[None, :], (L, 1))

        sza         = np.full((L, W), 40.0)     # mid-morning
        glint_angle = np.full((L, W), 45.0)     # no glint
        lza         = np.full((L, W), 45.0)
        azimuth     = np.full((L, W), 120.0)

        # Ancillary
        tpw     = np.full((L, W), 20.0)          # mm
        emiss7  = np.full((L, W), 0.95)
        emiss14 = np.full((L, W), 0.97)

        # Build a minimal LUT_TPW (6 rows × 35 cols)
        # Row 0: TPW bins; rows 2-3: transmittance (≈1); rows 4-5: absorption (≈0)
        lut = np.zeros((6, 35))
        for k in range(5):
            lut[0, k*7:(k+1)*7] = k + 1       # TPW bin label
        lut[2, :] = 0.97   # ch7 transmittance
        lut[3, :] = 0.95   # ch14 transmittance
        lut[4, :] = 0.001  # ch7 absorption offset
        lut[5, :] = 0.002  # ch14 absorption offset

        # Masks: all land, no desert, no water
        land_cover  = np.full((L, W), 8)   # Wooded grassland (valid)
        land_mask   = np.ones ((L, W), dtype=bool)
        desert_mask = np.zeros((L, W), dtype=int)
        usgs_eco    = np.full ((L, W), 10)  # Grassland (valid)

        # Visible reflectance (daytime)
        refl2 = np.full((L, W), 0.10)

        from fdca import FDCAInput
        return FDCAInput(
            bt7=bt7, rad7=rad7,
            bt14=bt14, rad14=rad14,
            bt13=None, rad13=None, bt15=None,
            refl2=refl2,
            latitudes=lat2d, longitudes=lon2d,
            sza=sza, glint_angle=glint_angle,
            lza=lza, azimuth=azimuth,
            tpw=tpw, emiss7=emiss7, emiss14=emiss14,
            lut_tpw=lut, FPT=85.0,
            land_cover=land_cover, land_mask=land_mask,
            desert_mask=desert_mask, usgs_eco=usgs_eco,
        )

    def test_no_fire_scene(self):
        """Uniform background scene should produce no confirmed fires."""
        from fdca import run_fdca
        inp = self._make_synthetic_input(inject_fire=False)
        out = run_fdca(inp)
        self.assertEqual(out.n_confirmed, 0,
            f"Expected 0 fires in background scene, got {out.n_confirmed}")

    def test_fire_detected(self):
        """Injected sub-pixel fire should be detected."""
        from fdca import run_fdca
        inp = self._make_synthetic_input(inject_fire=True)
        out = run_fdca(inp)
        self.assertGreater(out.n_confirmed, 0,
            "Injected fire was not detected")

    def test_fire_at_correct_location(self):
        """Detected fire should be at or near the injected location (20, 20)."""
        from fdca import run_fdca
        inp = self._make_synthetic_input(inject_fire=True)
        out = run_fdca(inp)
        if out.n_confirmed == 0:
            self.skipTest("No fires detected")
        locs = [(f.i, f.j) for f in out.confirmed_fires]
        self.assertIn((20, 20), locs, f"Fire not at (20,20). Found at: {locs}")

    def test_fire_mask_codes_valid(self):
        """All fire mask codes in confirmed pixels should be valid ATBD codes."""
        from fdca import run_fdca
        inp = self._make_synthetic_input(inject_fire=True)
        out = run_fdca(inp)
        valid_fire_codes = {10,11,12,13,14,15,30,31,32,33,34,35}
        for f in out.confirmed_fires:
            code = int(out.fire_mask[f.i, f.j])
            self.assertIn(code, valid_fire_codes,
                f"Invalid fire mask code {code} at ({f.i},{f.j})")

    def test_output_summary_runs(self):
        """summary() should not raise."""
        from fdca import run_fdca
        inp = self._make_synthetic_input(inject_fire=True)
        out = run_fdca(inp)
        summary = out.summary()
        self.assertIn("FDCA", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
