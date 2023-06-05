# -*- coding: utf-8 -*-
import os
import re
from collections import OrderedDict

import matplotlib.pyplot as plt

iter_pat = re.compile("(?<=iter:)(.*)(?=, lr)")
epoch_pat = re.compile("(?<=\]\[epoch:)(.*)(?=, iter:)")
loss_pat = re.compile("(?<=l_pix:)(.*)")
val_pat = re.compile("(?<=# psnr:)(.*)(?=Best: )")
log_time = re.compile("(?<=_........_)(.*)(?=.log)")
log_day = re.compile("(?<=_)(........)(?=_)")


epoch_vals = []
epoch_vals_dict = OrderedDict()
epoch_loss_dict = OrderedDict()

epoch_now = 0
iter_now = 0
iter_nums = []
loss_nums = []

val_iters = []
val_nums = []
log_path = os.listdir('/home/lab/code1/IR/experiments/train_MYIR_scratch/')
# log_path = os.listdir('/root/autodl-tmp/pycharm_project_983/experiments/train_MYIR_scratch/')
log_path = list(filter(lambda x:x.__contains__('.log'), log_path))
def get_time(name):
    return int(log_time.search(name)[0])+int(log_day.search(name)[0])*1000000
log_path.sort(key=get_time)

for log_f in log_path:
    print(log_f)
    # with open('/root/autodl-tmp/pycharm_project_983/experiments/train_MYIR_scratch/'+log_f, 'r') as f:
    with open('/home/lab/code1/IR/experiments/train_MYIR_scratch/'+log_f, 'r') as f:
        for line in f:
            if line.__contains__('[epoch'):
                # print(line)
                iter_now = int(iter_pat.search(line)[0].replace(',', ''))
                loss = float(loss_pat.search(line)[0].replace(',', ''))
                epoch_now = int(epoch_pat.search(line)[0].replace(',', ''))

                epoch_loss_dict[epoch_now] = loss
                loss_nums.append(loss)
                iter_nums.append(iter_now)
            elif line.__contains__('# psnr:'):
                val_iters.append(iter_now)
                val_nums.append(float(val_pat.search(line)[0].replace(',', '')))
                epoch_vals_dict[epoch_now] = float(val_pat.search(line)[0].replace(',', ''))

# plt.plot(iter_nums, loss_nums)
# plt.show()
plt.plot(epoch_vals_dict.keys(), epoch_vals_dict.values())
# plt.plot(epoch_loss_dict.keys(), epoch_loss_dict.values())
plt.show()
a= 1
