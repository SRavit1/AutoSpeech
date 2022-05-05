#Source of data: https://docs.google.com/spreadsheets/d/1zUbL2iqfdpDMZrpLPLzKySnAJ7A8FgEw0xEauqdjswQ/edit#gid=0
import os
import numpy as np
from matplotlib import pyplot as plt

log_dir = "../logs/autospeech"
eer_vs_runtime_file = "eer_vs_runtime_plot.png"
eer_vs_runtime_path = os.path.join(log_dir, eer_vs_runtime_file)
eer_vs_parameters_file = "eer_vs_parameters.png"
eer_vs_parameters_path = os.path.join(log_dir, eer_vs_parameters_file)

eer_vals = [10.2, 12.3, 11.99, 9.13, 9.01, 9.05, 9.72, 13.6, 14.1, 13.9, 9.3, 12.54, 19.49]
mu_runtime = [228812224.3, 373266741.2, 746533482.5, 159515701.4, 574256525, 574256525, 373266741.2, 41095221829, 2568451364, 642112841.1, 160528210.3, 80264105.13, 40132052.57]
result_ids = ["Paper_VGGM", "Paper_ResNet18", "Paper_ResNet34", "Paper_Proposed1", "Paper_Proposed2", "Paper_Proposed3", "full_precision", "fp_quant_32/32", "fp_quant_8/8", "fp_quant_4/4", "fp_quant_2/2", "fp_quant_2/1", "fp_quant_1/1"]
storage_kb = [1572000, 46800, 86400, 20000, 72000, 72000, 46800, 46800, 11700, 5850, 2925, 1462.5, 1462.5]

modified_result_ids = ["/".join(result_id.split("_")) for result_id in result_ids]

to_remove = [9, 8, 7, 5, 4, 3] #first 3- results outdated, many things probably changed; second 3- unsure of runtime
to_remove.sort(reverse=True)
for i in to_remove:
  del eer_vals[i]
  del mu_runtime[i]
  del modified_result_ids[i]
  del storage_kb[i]

fig, ax = plt.subplots()
ax.scatter(mu_runtime, eer_vals)
ax.set_title("Speaker Verificaiton EER vs Runtime")
ax.set_xlabel("Runtime (mu_s)")
ax.set_ylabel("EER")

for i in range(len(modified_result_ids)):
  result_id = modified_result_ids[i]
  runtime = mu_runtime[i]
  eer = eer_vals[i]
  ax.annotate(result_id, (runtime, eer+0.1))
fig.savefig(eer_vs_runtime_path, bbox_inches='tight')
plt.close('all')

to_remove = [0] #removing VGG because # parameters is outlier
to_remove.sort(reverse=True)
for i in to_remove:
  del eer_vals[i]
  del mu_runtime[i]
  del modified_result_ids[i]
  del storage_kb[i]

fig, ax = plt.subplots()
ax.scatter(storage_kb, eer_vals)
ax.set_title("Speaker Verificaiton EER vs Parameter Storage")
ax.set_xlabel("Parameter Storage Size (KB)")
ax.set_ylabel("EER")
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

for i in range(len(modified_result_ids)):
  result_id = modified_result_ids[i]
  parameter_size = storage_kb[i]
  eer = eer_vals[i]
  ax.annotate(result_id, (parameter_size, eer+0.1))
fig.savefig(eer_vs_parameters_path, bbox_inches='tight')

plt.close('all')
