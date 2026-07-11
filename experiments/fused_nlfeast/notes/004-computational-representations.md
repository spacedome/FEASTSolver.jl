# 004: Computational Representations

Question: can corrected moments retain the SS projection-width tradeoff, and
can the common overlap avoid m² unreduced divided-difference matrices?

Tangential compression is an exact input projection: Qₖ becomes QₖΩ. It
preserves the pole state when the compressed realization is controllable. On
the seven-root scalar control, full width seven gives a 7-by-49 corrected
Hankel and converges in two updates. Width one gives 7-by-7 and takes three;
width three gives 7-by-21 and again takes two. For a repeated 2+1 spectrum,
depth one requires width three. At depth two, width two retains the complete
state and converges, but needs one additional update. The implementation keeps
compression opt-in and rejects widths violating d·p≥m or detected cluster
multiplicity. Sparse partitioned tangents retain these results with one
nonzero per state row, avoiding a dense n-by-m-by-p contraction.

The preferred nonlinear primitive is now the small tangential matrix G. Exact
polynomial assembly agrees with operator-valued divided differences to about
1e-15. Residual completion needs true divided differences only for 3/9 entries
on a simple three-root problem, 5/9 for a repeated 2+1 problem, 7/49 for scalar
sine, and 8/36 for the nonnormal repeated cluster.

An illustrative dense n=384, m=28 control took 0.85 seconds and allocated
1.73 GiB when copying and applying a full divided-difference matrix for every
entry. Structured overlap assembly took 0.0018 seconds and 0.18 MiB; residual
completion including block residual applications took 0.009 seconds and
1.53 MiB. These are shape diagnostics, not package benchmarks.

Decision: support explicit tangential moment widths and problem-level
`divided_overlap` assembly. Keep uncompressed moments as the default until
count completeness is implemented. Keep residual completion available for
generic operators, but do not make its cluster tolerance implicit policy.
