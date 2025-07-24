# 该脚本执行训练和测试过程，结束后发送邮件提醒
import time
import subprocess


subprocess.run(["python", "pre_process.py"])
subprocess.run(["python", "get_graph.py"])
subprocess.run(["python", "train2.py"])
subprocess.run(["python", "get_metrics2.py"])
sent_email()
