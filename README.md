## Method recommendation: stratified Monte Carlo

- enumerate the cartesian product of all multi-selects → strata
- weight each stratum uniformly (1 / total strata)
- inside each stratum, LHS-sample the continuous inputs (~50 samples)
- apply the cost formula per sample
- pool weighted samples across strata, sort, read percentiles
- output is one `F_mix(c)`, one band
---

### Quick note
**You will come across the term LHS**, it's meaning is:
It is a sampling method used as an alternative to pure random Monte Carlo sampling. For this project, it is used specifically inside each discrete "stratum" to sample continuous inputs (like concurrency, tokens, and utilization).

---
## Inputs come from the UI as

- multi-select dropdowns for discrete (GPU, model, hosting)
- point + range pairs for continuous (concurrency, complexity, utilization)
- weights for the discrete options are not captured anywhere, so default to uniform 1/N over the N selected options

## Why this beats plain Monte Carlo

**Accuracy.** Stratified sampling has provably lower variance than IID sampling when the stratification variable correlates with the output. This is a textbook result. The discrete inputs (GPU, hosting, model) swing cost significantly. GPU choice alone can move cost by 3-5×, hosting type by even more. So they are high-leverage stratifiers. By enumerating them exactly with uniform weights, we eliminate the binomial wobble that plain MC has from sampling the discrete dimension. Inside each stratum, LHS adds another 4-10× variance reduction over pure random sampling at the same N. Net effect: at equal compute, stratified MC gives tighter, more stable percentile estimates than plain MC. Or equivalently, ~5-10× fewer samples for the same precision.

**Feasibility.** Plain MC is ~10 lines of numpy. Stratified MC is ~40 lines: an enumeration loop over discrete combos, LHS sampler from `scipy.stats.qmc`, and a weighted-percentile combine at the end. Both well within a day of implementation work. Small overhead, substantial accuracy gain.

**Bonus.** Each stratum is already a conditional cost distribution, so compare-by-GPU and compare-by-hosting views come for free. We won't just recombine plot per-stratum bands directly.

## Why pure analytical isn't feasible

The cost function in `gpu_capacity.total_gpus_throughput_based` uses `math.ceil` to compute GPU counts. Step functions don't compose cleanly through analytical CDFs. Could be linearized but loses accuracy in some regimes. Stratified MC sidesteps this by sampling inside each stratum.

## Real-world walkthrough: AgriBest use case

User's inputs from the UI:

- Users: range 200 to 500 (continuous, becomes the x-axis sweep)
- Complexity: medium tier, tokens 8k to 12k per request (uniform range)
- Concurrency: 15 to 25 percent (uniform range)
- Utilization: 60 to 80 percent (uniform range)
- GPUs: multi-select `[A100, H100]`
- Hosting: multi-select `[Cloud, API]`
- Model: Llama 8B (single)

Goal: produce the cost band across user counts 200 to 500.

### Step 1, enumerate strata from discrete multi-selects

```
GPUs: [A100, H100]        → 2 options
Hosting: [Cloud, API]     → 2 options
Model: [Llama 8B]         → 1 option (no contribution)
Cartesian product         → 4 strata:
  (A100, Cloud, Llama 8B)
  (A100, API,   Llama 8B)
  (H100, Cloud, Llama 8B)
  (H100, API,   Llama 8B)
Weight per stratum        → 1/4 = 0.25
```

### Step 2, pick the first x-value on the chart

Say `x = 300 users`.

### Step 3, for each stratum at x = 300, sample the continuous inputs jointly

Take stratum `(A100, Cloud, Llama 8B)`. The discrete part is now fixed, so the cost formula has fixed GPU specs (A100 memory and throughput) and fixed pricing structure (Cloud hourly rates). Generate 50 LHS samples, each one a complete tuple `(concurrency_i, tokens_i, utilization_i)`:

```
sample 1:  concurrency=0.16, tokens=8400,  utilization=0.62
sample 2:  concurrency=0.18, tokens=9100,  utilization=0.74
sample 3:  concurrency=0.21, tokens=10200, utilization=0.66
...
sample 50: concurrency=0.24, tokens=11800, utilization=0.78
```

This is where the normals and uniforms combine. Each sample is one independent draw from each input distribution. Three independent input distributions become one joint sample. LHS just makes the joint coverage of the 3D space more even than pure random.

### Step 4, apply the cost formula to each sample

```
cost(users=300, concurrency=0.16, tokens=8400, utilization=0.62, A100, Cloud, Llama 8B)
  = throughput needed → GPUs needed (ceiling) → hourly rate × hours
  = $X_1

cost(users=300, concurrency=0.18, tokens=9100, ...) = $X_2
...
cost(users=300, concurrency=0.24, tokens=11800, ...) = $X_50
```

Result: 50 cost values for stratum `(A100, Cloud)` at x = 300. This collection IS the empirical conditional cost distribution `F_{A100,Cloud}(c | users=300)`.

### Step 5, do the same for the other 3 strata

Get 50 cost samples each for `(A100, API)`, `(H100, Cloud)`, `(H100, API)`. Now you have 200 cost values total at x = 300, each tagged with its stratum.

### Step 6, combine into one mixture distribution at x = 300

Since each stratum has equal weight (0.25) and equal sample count (50), pooling all 200 samples with weight 1/200 each is correct. Sort the 200 samples by cost. Read positions 20, 100, and 180 for P10, P50, P90. Those are the three band values at x = 300.

If sample counts per stratum varied, you'd use weighted percentiles instead.

### Step 7, repeat steps 2 to 6 across x-values

```
x = 200 users → (P10, P50, P90) = ($a₁, $b₁, $c₁)
x = 215 users → ($a₂, $b₂, $c₂)
x = 230 users → ($a₃, $b₃, $c₃)
...
x = 500 users → ($a₂₀, $b₂₀, $c₂₀)
```

Plot the three series. The P50 line is your central estimate, the P10 and P90 lines bound the band. That's the chart.

## Where each input type lives in this pipeline

| Input type | Example | Where it lives |
|---|---|---|
| Discrete multi-select | GPUs: [A100, H100] | Becomes the strata. Weights from cardinality |
| Continuous (sweep) | Users: 200-500 | Becomes the x-axis. We loop over x |
| Continuous (uniform) | Utilization: 60-80% | Sampled inside each stratum |
| Continuous (normal) | Concurrency: ~20% ± 5% | Sampled inside each stratum |
| Continuous (uniform) | Complexity tokens: 8k-12k | Sampled inside each stratum |
| Single (no uncertainty) | Model: Llama 8B | Fixed constant in every stratum |

The discretes get exact handling via enumeration. The sweep variable gets exact handling via looping. The other continuous inputs get sampled jointly inside each stratum. Single-option inputs are just constants.

**Total compute for one chart**: 20 x-values × 4 strata × 50 LHS samples = 4,000 cost evaluations. Python does that in well under a second.

## Pipeline diagram

<img width="1840" height="3035" alt="stratified mc" src="https://github.com/user-attachments/assets/6c499026-ffbc-4037-bc94-691c30185c3e" />

