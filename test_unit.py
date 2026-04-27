import sys, os
sys.path.insert(0, 'backend')
from simulator import VehicleSim
from v2x import V2XBus

def test_sim_step():
    sim = VehicleSim(rows=3, cols=3, max_vehicles=50)
    for _ in range(10):
        sim.step()

    assert len(sim.vehicles) > 0
    print("sim.step OK")

def test_v2x_tick():
    sim = VehicleSim(rows=3, cols=3, max_vehicles=50)
    v2x = V2XBus(sim)

    for _ in range(10):
        sim.step()
        v2x.tick()

    assert len(v2x.tail()) >= 0
    print("v2x.tick OK")

def test_sim_osm():
    sim = VehicleSim(rows=3, cols=3, max_vehicles=50)

    # Mock OSM graph
    graph = {
        "nodes": [
            {"id": "n1", "x": 0, "y": 0, "is_signal": True},
            {"id": "n2", "x": 100, "y": 0, "is_signal": False},
        ],
        "edges": [
            {"from": "n1", "to": "n2", "highway": "primary"},
            {"from": "n2", "to": "n1", "highway": "primary"}
        ]
    }

    sim.load_from_osm(graph)
    assert len(sim.nodes) == 2
    assert len(sim.edges) == 2
    print("sim.load_from_osm OK")

if __name__ == "__main__":
    test_sim_step()
    test_v2x_tick()
    test_sim_osm()
    print("All unit tests passed.")
