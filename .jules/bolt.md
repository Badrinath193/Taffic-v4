## 2025-04-29 - Pre-computing edge geometry and short-circuiting distance checks
**Learning:** In hot loops like per-vehicle simulator updates, repeatedly fetching node coordinates to check edge orientation, and invoking functions before cheap distance checks can be a significant bottleneck.
**Action:** Pre-compute static geometry data (like `horizontal`) on edge creation and use short-circuiting (`dist < 14.0 and not self._is_green_for_edge`) to skip function calls for vehicles far from intersections.
