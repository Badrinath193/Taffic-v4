import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from simulator import VehicleSim
from v2x import V2XBus

def main():
    sim = VehicleSim(rows=5, cols=5, max_vehicles=1000)
    v2x = V2XBus(sim)

    # Warmup
    for _ in range(50):
        sim.step()
        v2x.tick()

    start_time = time.time()
    for _ in range(2000):
        sim.step()
        v2x.tick()
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"Elapsed time for 2000 steps with V2X: {elapsed:.4f}s")
    print(f"Steps per second: {2000 / elapsed:.2f}")

if __name__ == "__main__":
    main()
