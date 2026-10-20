Fitted on the 1200 MHz run: sustained_sm_clock_mhz = 1834, bandwidth_knee_sm_clock_mhz = 1564. Unlocked clock identical to no clock: True.

| variant | prefill | decode step | end-to-end |
|---|---:|---:|---:|
| full-clock baseline vs full-clock measured | 4.3% | 0.5% | 1.5% |
| naive | 13.8% | 22.8% | 9.6% |
| compute-only | 4.8% | 22.8% | 14.0% |
| clock-fit | 4.8% | 3.0% | 2.6% |
| clock-loo | 4.9% | 4.3% | 3.2% |
| clock-fit-curve | 1.1% | 3.0% | 2.6% |

| shape | naive prefill / decode | clock-fit prefill / decode | clock-loo prefill / decode |
|---|---|---|---|
| 1x512 | -3.9% / -20.0% | -16.1% / +4.3% | -16.2% / +5.8% |
| 1x2048 | +13.7% / -19.6% | -0.7% / +4.8% | -0.8% / +6.2% |
| 8x512 | +14.8% / -25.4% | +0.3% / -2.8% | +0.6% / -4.3% |
| 8x2048 | +18.0% / -25.7% | +3.0% / -3.2% | +3.2% / -4.8% |
| 32x512 | +18.7% / -23.3% | +3.7% / -0.1% | +3.9% / -0.3% |
