# 009: State Selection

State extraction must realize broadly before selecting the chart. Loaded
string often has right/left total ranks `(5,6)` while both sides contain the
same five interior roots. Requiring total-rank equality caused false aborts;
allowing independent total ranks and comparing interior counts repaired every
24- to 64-node seed.

Right and left Schur restriction are dual operations. Selected right states
must lead the ordered Schur form; selected left states must trail it after the
unselected states are ordered first. Taking the leading block on the left was
an algebraic bug: the omitted Schur coupling remained in the left invariant
equation. It caused the loaded-string left residual to stall near `10⁻²`.

After the correction, the common Schur state reaches two-sided backward errors
below `10⁻¹⁰` on every loaded-string seed with a complete initial rank, at 16,
24, and 32 nodes. The earlier evidence that modal updates were substantially
faster was therefore invalid. A Schur invariant pair is the canonical state;
diagonal modal form is an optional optimization for separated blocks.

Initial Schur restriction and later continuation can still differ near an
unresolved boundary. Fixed-rank, chart-restricted, and rollback candidates are
cheap because they reuse moments, but none can replace count and quadrature
refinement when the initial realization is deficient.

Common balancing may also move an otherwise interior independent realization
onto a neighboring exterior root. Common and fixed-rank candidates are now
gated by final chart containment, and count matching requires both state
spectra to stay inside. An exterior continuation is recorded only as a rejected
candidate; it is never allowed to steer the persistent state.

One additional random probe column repairs every deficient loaded-string seed.
Immediate tangent compression to width two is not reliable even when the
small-state controllability test passes. A compressed candidate must contract
the two-sided residual or be reassembled at full width from the same cached
solves. Compression changes moment storage and reduced work, not the number of
large residual-basis solves.

One cross-Hankel pencil yields the common state, right map, and dual left map
in the continuous-moment limit. Finite quadrature introduces a left filter
factor. Divided-overlap balancing improves this direct realization, but the
loaded-string left side still converges more slowly than two independently
observable realizations followed by common gauging. The one-pencil form remains
a useful diagnostic candidate rather than the primary numerical update.

Common divided-overlap coupling is exact algebraically. Numerically it is an
optional candidate gated by overlap conditioning and two-sided backward error.
The arithmetic average of its two action formulas only symmetrizes roundoff;
the formulas are algebraically identical.
