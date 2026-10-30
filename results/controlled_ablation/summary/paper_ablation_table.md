# Controlled forward-model ablation

All variants have identical registered parameter counts and use paired seeds.

| Variant | Seeds | Parameters | Q_sca MAE (mean ± SD) | RMSE (mean ± SD) | R² (mean ± SD) |
|---|---:|---:|---:|---:|---:|
| raw_k0 | 5 | 1397762 | 0.00885444 ± 0.00128327 | 0.02167148 ± 0.00384088 | 0.99951426 ± 0.00017691 |
| fourier_only | 5 | 1397762 | 0.00910900 ± 0.00045693 | 0.01999144 ± 0.00102982 | 0.99959593 ± 0.00004132 |
| physics_only | 5 | 1397762 | 0.00791426 ± 0.00033971 | 0.02126073 ± 0.00067377 | 0.99954360 ± 0.00002869 |
| fourier_physics_additive | 5 | 1397762 | 0.00823573 ± 0.00046994 | 0.01944820 ± 0.00183911 | 0.99961567 ± 0.00007386 |
| full_pgff | 5 | 1397762 | 0.00754496 ± 0.00040291 | 0.01760385 ± 0.00039325 | 0.99968722 ± 0.00001390 |
