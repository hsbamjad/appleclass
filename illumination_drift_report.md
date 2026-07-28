# Illumination Drift Study: Halogen vs LED
**MSU Apple Classification System | JAI FS-3200T Camera**
**Date: 2026-07-28 | Purpose: Identify source of temporal illumination drift**

---

## Background

During formal apple data collection runs, significant lamp brightness drift was
observed between calibration captures and apple run captures. Scale factors computed
by script c_ ranged from 0.449 to 0.504 for RGB, indicating the lamp was 5 to 12
percent brighter at run time than at calibration. This raised the question: is the
halogen lamp the source of drift, or is something else responsible?

A controlled test was conducted to isolate the illumination source as the variable.

---

## Test Method

A Spectralon white reference panel (75% reflectance) was placed at the center
position on the conveyor belt. A single camera channel (RGB, ch1) captured the
panel under two illumination conditions: LED only, then halogen only. Locked AWB
was applied throughout. The panel center brightness (20% center crop, grayscale
mean DN) was recorded at multiple time intervals from cold start.

All readings used the same panel position, same camera, and same image analysis
script. Only the illumination source changed between the two test conditions.

---

## Results

### LED Illumination (halogen OFF)

| Reading | Time (min) | Gray DN | Change vs t=0 |
|---------|-----------|---------|--------------|
| 1 | 0 | 47.02 | baseline |
| 2 | 2 | 47.18 | +0.3% |
| 3 | 7 | 47.35 | +0.7% |
| 4 | 22 | 48.00 | +2.1% |
| 5 | 30 | 46.86 | -0.3% |

Total variation over 30 minutes: -0.3% to +2.1% (range of 2.4%)

LED output increased slightly in the first 22 minutes (thermal warm-up of the LED
driver), then settled back near baseline by t=30. The drift was not monotonic and
stayed within a narrow band throughout the session. No reading deviated more than
1 DN from the mean.

### Halogen Illumination (LED OFF)

| Reading | Time (min) | Gray DN | Change vs t=0 | Step change |
|---------|-----------|---------|--------------|------------|
| 1 | 0 | 156.38 | baseline | -- |
| 2 | 7 | 148.75 | -4.9% | -7.63 DN |
| 3 | 12 | 142.69 | -8.8% | -6.06 DN |
| 4 | 18 | 142.12 | -9.1% | -0.57 DN |
| 5 | 28 | 153.56 | -1.8% | +11.44 DN |
| 6 | 30 | 146.31 | -6.4% | -7.25 DN |
| 7 | 32 | 158.95 | +1.6% | +12.64 DN |
| 8 | 33 | 141.59 | -9.5% | -17.36 DN |
| 9 | 35 | 160.74 | +2.8% | +19.15 DN |
| 10 | 40 | 164.16 | +5.0% | +3.42 DN |

Peak-to-peak range over 40 minutes: 141.59 to 164.16 = 22.57 DN = **14.4% of baseline**

The halogen output is non-monotonic and oscillatory with no convergence observed.
Between reads 8 and 9 (2 minutes apart), output jumped 19.15 DN -- a single-step
change of 12.1%. Between reads 7 and 8 (1 minute apart), it dropped 17.36 DN.
These short-interval swings are not thermal drift -- they indicate active instability
in the lamp-ballast system. The lamp cannot be characterized by any fixed correction
factor, and its output cannot be predicted without a live reference measurement.

---

## Comparison

| Metric | LED | Halogen |
|--------|-----|---------|
| Drift at 7 min | +0.7% | -4.9% |
| Drift at 22-25 min | +2.1% | -9.1% |
| Drift at 30-35 min | -0.3% | -1.8% to +2.8% |
| Peak-to-peak range (full session) | 2.4% | 14.4% |
| Max single step change | 1.14 DN / 2.4% | 19.15 DN / 12.1% |
| Stabilization observed | Yes (by t=30) | No (still oscillating at t=40) |
| Behavior pattern | Narrow oscillation, self-correcting | Wide oscillation, unpredictable |

---

## Conclusion

The halogen lamp is confirmed as the source of temporal illumination drift in the
apple classification system. LED illumination produces only 2.4% variation over a
30-minute session. Halogen produces 14.4% peak-to-peak swing over 40 minutes with
individual step changes of up to 12% occurring within single minutes.

The halogen behavior is not a simple warm-up curve but active oscillation caused by
the interaction between tungsten filament thermal mass, lamp housing convection, and
power supply regulation. Because the lamp can swing 10-12% in either direction within
2 minutes, no single correction factor applied at session start can account for its
behavior during the session.

These findings validate the per-run panel calibration approach implemented in script
c_: capturing the white reference panel immediately before each apple run measures
the lamp state at that exact moment, allowing an accurate scale factor to be computed
regardless of where the lamp is in its oscillation cycle.

**For future data collection:** LED-only illumination would reduce drift by 6x and
eliminate the need for per-run correction on RGB channels. However, NIR channels
require halogen due to the LED spectral gap at 800-900 nm. The current per-run panel
approach remains the correct solution for the halogen-dependent NIR channels.

---

*Test conducted: 2026-07-28 | Analyst: MSU Apple Classification Project*
*Chart: drift_comparison_chart.png (same folder)*
