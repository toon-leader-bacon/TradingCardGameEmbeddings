# evaluation

Post-training evaluation of encoder checkpoints, run offline by hand from
scripts. **Intrinsic** evaluation fits no model: it embeds cards once into a
table, then analyzes the stored vectors (projection plots, cluster
agreement, label compactness). **Extrinsic** evaluation trains fresh dojo
heads on a frozen encoder by reusing `Trainer`, and compares the loss
curves. `Trainer` never calls this package.

Status: planned, no code yet. The design lives in
[`plans/evaluation.md`](../../plans/evaluation.md).
