# Findings

The cache-native idea is valid as an execution and policy architecture. It is
not one mandatory update formula beyond the corrected-moment master form.

```text
mandatory consumer     corrected moment construction
conditional consumer   residual-forced spectral repair
conditional consumer   closure/leakage-forced response
persistent data        physical state and bounded moment space
generation-local data  numerical factors and sampled responses
```

| Question | Evidence | Conclusion |
|---|---|---|
| Can one factor cache supply extraction and response? | Hermitian cached extraction and response match standalone paths exactly | Yes |
| Does the cache agree with fused contour-sample algebra? | Bridge moment sequences agree exactly | Yes; factor and sample caches are separate layers |
| Are explicit corrected moments needed for `zI−H(θ)`? | Explicit and FEAST-identity moments agree near roundoff; identity uses `Nℓ` instead of `Np` RHS | No; specialize the linear pencil |
| Can response reduce refactorization rounds? | Four-block Hermitian control falls 12→9 refreshes; dual constrained control 16→7 | Yes, conditionally |
| Should response always be used? | Adequately solved windows gain nothing; sparse growing moments remain faster | No |
| Should a general NEP be spectrally repaired repeatedly? | Extra quadratic repairs lower spectral residual but not six closure refreshes | Only under residual forcing |
| Does residual compression preserve correction? | Rank-one control matches uncompressed moments and reduces width `p→1` | Yes |
| Can dual factors serve both sides and response? | Dual response matches finite differences and shares one LU | Yes |
| Is full-projector response structurally prohibitive? | Nonlocal action matches finite differences, costs `Np` RHS, and has rank ≤`2p` | Not per action; low-rank Krylov remains open |
| Does a factor/RHS ratio select the winner? | Kernel ratio predicts a response policy that loses the end-to-end sparse timing | Not alone |

The current candidate schedule is:

```text
freeze θ and build a chart generation
→ form the corrected moment block
→ append it to a bounded pre-extraction window
→ repair the spectral pair only until its residual forcing target
→ spend the cache on response only when the prior phase was closure-limited
  and the measured cost model permits it
→ invalidate factors on the first accepted θ change
→ continue reduced closure iteration until closure reaches the leakage floor
→ refresh
```

Periodic response remains an experimental upper envelope because it sometimes
beats the defect/leakage trigger. The adaptive trigger is the natural safe
policy but lacks a contraction predictor. A production implementation should
measure factorization, block solves at candidate widths, reduced operator
builds, and synchronization, then choose among no response, one response, or
another reduced step.

The following alternatives are rejected as defaults:

- carrying numerical factors after changing the nonlinear state;
- carrying unsafeguarded multisecant history across changed reduced maps;
- explicit residual correction for a linear pencil;
- fixed repeated spectral repair;
- response on every refresh;
- unbounded moment growth as the production memory policy;
- selecting a policy from factor/RHS counts without reduced and communication costs.

The experiment does not establish GPU speedup, general `T(θ,z)` response,
low-rank projector Krylov, or convergence theory. `CACHE_CONTRACT.md` states the
interfaces and mathematical conditions those follow-on projects must preserve.
