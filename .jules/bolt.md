## 2024-05-05 - Simulator Loop Optimization
**Learning:** In highly iterated loops like per-vehicle traffic simulation, recalculating static geometry (e.g. edge direction/horizontality) and fetching node coordinates per-frame causes noticeable performance degradation. Distance checks that might short-circuit expensive logic are critical.
**Action:** Always pre-compute or lazy-cache static geometry properties rather than deriving them from nodes in hot loops. Rearrange conditions to short-circuit expensive functions.
