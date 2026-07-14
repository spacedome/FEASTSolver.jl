# Nonlocal Projector Control

The local-density controls do not prove that the schedules work when the
Hamiltonian needs off-diagonal projector information. The separate control

```text
H(P) = H₀ + g K P K,    P = XXᴴ
```

uses a normalized Gaussian convolution `K`. Two projectors can have identical
diagonals and produce different Hamiltonians, so this problem cannot be routed
through the density implementation.

At `n=40,p=3,g=4.2`, full diagonalization with projector mixing converges at
mixing `0.1` but fails at `0.2` and above. With projector-space Anderson in the
reduced solve, the retained FEAST schedules give:

```text
fixed q=5,6,8,10       14, 11, 9, 8 refreshes
three-block window      7 refreshes at 5 inner updates
four-block window       6 refreshes at 5 inner updates
growing memory          6 refreshes at 5 inner updates
```

Adaptive thick restart starting at `q=p=3` grows to `q=6` and converges. The
same qualitative hierarchy as the local-density problem therefore survives a
genuinely nonlocal, gauge-invariant projector closure.

The inner update matters. Plain projector mixing needs roughly 170–220 reduced
iterations on this control. Projector-space Anderson, with Hermitian and trace
preservation, reduces that by several times. The final state is a projector to
the requested tolerance; intermediate Anderson states are trace-corrected
Hermitian density matrices but are not explicitly clipped to `[0,I]`.
