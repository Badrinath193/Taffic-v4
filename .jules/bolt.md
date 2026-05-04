# Bolt Journal
## 2024-05-24 - Short-circuiting distance checks in `simulator.py`
**Learning:** Found a hot loop bottleneck in the `VehicleSim.step` method. Evaluating `_is_green_for_edge` for every vehicle in every tick is computationally expensive, especially when the check isn't relevant to vehicles that are far away from an intersection.
**Action:** Adding a simple short-circuit `if dist_to_end < 14.0:` before the `_is_green_for_edge` call reduced simulation time by ~25% in tests.
