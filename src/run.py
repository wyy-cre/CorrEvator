import time
import subprocess


start = time.time()
subprocess.run(["python", "pre_process.py"])
subprocess.run(["python", "get_graph.py"])
subprocess.run(["python", "train.py"])
subprocess.run(["python", "get_metrics.py"])

print(f"总时常: {(time.time() - start) / 60} min")
