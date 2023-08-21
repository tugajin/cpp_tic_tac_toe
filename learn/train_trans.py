import subprocess
from merge_count import *
from merge_resolved import *
from pseudo_data import *
from train_network_trans import *
import sys

args = sys.argv
num = int(args[1])

merge_count()
#merge_resolved()
# if num == 1 or (num != 0 and num % 10 == 0):
#     print("pseudo")
#     preudo_data()
train_network()

