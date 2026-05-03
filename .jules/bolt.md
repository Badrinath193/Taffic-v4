
## $(date +%Y-%m-%d) - Precompute static edge properties
**Learning:** Checking edge geometry (e.g. `horizontal`) dynamically during the simulation hotloop (`VehicleSim.step()`) added significant overhead, despite being static properties.
**Action:** When working on tight simulation loops, precompute all static network properties during initialization or graph construction (like `_build_grid` or `load_from_osm`) to avoid recomputing them dynamically.
