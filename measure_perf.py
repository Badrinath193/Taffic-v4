import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from simulator import VehicleSim

def main():
    sim = VehicleSim(rows=5, cols=5, max_vehicles=1000)

    # Warmup
    for _ in range(50):
        sim.step()

    start_time = time.time()
    for _ in range(2000):
        sim.step()
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"Elapsed time for 2000 steps: {elapsed:.4f}s")
    print(f"Steps per second: {2000 / elapsed:.2f}")

if __name__ == "__main__":
    main()
