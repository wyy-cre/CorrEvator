# 该脚本执行训练和测试过程，结束后发送邮件提醒
import time
import subprocess

point = time.time()
subprocess.run(["python", "pre_process.py"])
print(f"划分数据集耗时: {(time.time() - point) / 60} min")
point = time.time()
subprocess.run(["python", "get_graph.py"])
print(f"构图耗时: {(time.time() - point) / 60} min")
point = time.time()
subprocess.run(["python", "train.py"])
print(f"训练耗时: {(time.time() - point) / 60} min")
point = time.time()
subprocess.run(["python", "get_metrics.py"])
print(f"测试耗时: {(time.time() - point) / 60} min")
