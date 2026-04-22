import pytest
from backend.simulator import VehicleSim

def test_spawn_vehicle_limit():
    sim = VehicleSim(rows=1, cols=2, max_vehicles=2)
    # Simulator spawns randomly, so we can manually call _spawn_vehicle to test
    sim._spawn_vehicle()
    sim._spawn_vehicle()

    assert len(sim.vehicles) == 2

    # Try spawning more
    sim._spawn_vehicle()
    assert len(sim.vehicles) == 2

if __name__ == '__main__':
    pytest.main([__file__])
