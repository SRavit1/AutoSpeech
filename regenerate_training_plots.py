import json
import numpy as np
import os
from matplotlib import pyplot as plt
import sys

def generate_training_plots(config):
  fig_bw, ax_bw = plt.subplots()

  fig_loss, ax_loss = plt.subplots()
  fig_top1, ax_top1 = plt.subplots()
  fig_top5, ax_top5 = plt.subplots()
  fig_eer, ax_eer = plt.subplots()

  for (title, subdir) in config["models"].items():
    log_dir = os.path.join("../logs/autospeech", subdir)
    training_notes_path = os.path.join(log_dir, "metadata.json")

    with open(training_notes_path,'r') as f:
      training_data = json.load(f)
      full = training_data["type"] == "xnor"
      epochs = training_data["epochs"]
      abw_history = training_data["abw_history"][:len(epochs)]
      wbw_history = training_data["wbw_history"][:len(epochs)]
      loss_history = training_data["loss_history"]
      top1_history = training_data["top1_history"]
      top5_history = training_data["top5_history"]
      epochs_eval = training_data["epochs_eval"]
      eer_history = [elem*100 for elem in training_data["eer_history"]]

    bw_history = list(zip(abw_history, wbw_history))
    final_bw_epoch_index = bw_history.index(bw_history[-1])
    final_bw_epoch = epochs[final_bw_epoch_index]

    best_epoch = None
    lowest_eer = 0
    for eval_epoch_index, eval_epoch in enumerate(epochs_eval):
      if eval_epoch < final_bw_epoch:
        continue
      curr_eer = eer_history[eval_epoch_index]
      if best_epoch is None or curr_eer < lowest_eer:
        best_epoch = eval_epoch
        lowest_eer = curr_eer

    print(final_bw_epoch, best_epoch)

    if not full:
      ax_bw.plot(epochs, abw_history, label='activation bw')
      ax_bw.plot(epochs, wbw_history, label='weight bw')
      ax_bw.legend()
      ax_bw.vlines(final_bw_epoch, min(min(abw_history), min(wbw_history)), max(max(abw_history), max(wbw_history)), colors='r')

      plt.close('all')

    ax_loss.plot(epochs, loss_history, label=title)
    ax_loss.vlines(final_bw_epoch, min(loss_history), max(loss_history), colors='r')
    best_loss = loss_history[epochs.index(best_epoch)]
    #ax_loss.scatter([best_epoch], [best_loss], marker='x')
    #ax_loss.annotate("Best epoch: " + str(round(best_loss, 2)), (best_epoch, best_loss+0.1))

    ax_top1.plot(epochs, top1_history, label=title)
    ax_top1.vlines(final_bw_epoch, min(top1_history), max(top1_history), colors='r')
    best_top1 = top1_history[epochs.index(best_epoch)]
    #ax_top1.scatter([best_epoch], [best_top1], marker='x')
    #ax_top1.annotate("Best epoch: " + str(round(best_top1, 2)), (best_epoch, best_top1+0.1))

    ax_top5.plot(epochs, top5_history, label=title)
    ax_top5.vlines(final_bw_epoch, min(top5_history), max(top5_history), colors='r')
    best_top5 = top5_history[epochs.index(best_epoch)]
    #ax_top5.scatter([best_epoch], [best_top5], marker='x')
    #ax_top5.annotate("Best epoch: " + str(round(best_top5, 2)), (best_epoch, best_top5+0.1))

    epochs_eval = np.array(epochs_eval)
    eer_history = np.array(eer_history)
    eer_history_low = np.min(epochs_eval)
    eer_history_high = np.max(epochs_eval)
    indices = np.invert((eer_history==100))
    epochs_eval = epochs_eval[indices]
    eer_history = eer_history[indices]
    #ax_eer.set_xlim(eer_history_low, eer_history_high)
    ax_eer.plot(epochs_eval, eer_history, label=title)
    #ax_eer.plot(epochs_eval, np.array(eer_history))
    ax_eer.vlines(final_bw_epoch, min(eer_history), max(eer_history), colors='r')
    best_eer = eer_history[list(epochs_eval).index(best_epoch)]
    #ax_eer.scatter([best_epoch], [best_eer], marker='x')
    #ax_eer.annotate("Best epoch: " + str(round(best_eer, 2)), (best_epoch-15, best_eer+0.1))
  
  save_dir = config["save_dir"]
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  
  ax_bw.set_title("Bitwidth Adjustment")
  ax_bw.set_xlabel("Epochs")
  ax_bw.set_ylabel("Bitwidth")
  ax_bw.legend()
  fig_bw.savefig(os.path.join(log_dir, "bw_adjustment.png"))
  
  ax_loss.set_title("Training Loss Convergence")
  ax_loss.set_xlabel("Epochs")
  ax_loss.set_ylabel("Training Loss")
  ax_loss.legend()
  fig_loss.savefig(os.path.join(save_dir, "loss_convergence.png"))

  ax_top1.set_title("Training Top1 Accuracy Convergence")
  ax_top1.set_xlabel("Epochs")
  ax_top1.set_ylabel("Training Top1")
  ax_top1.legend()
  fig_top1.savefig(os.path.join(save_dir, "top1_convergence.png"))

  ax_top5.set_title("Training Top5 Accuracy Congergence")
  ax_top5.set_xlabel("Epochs")
  ax_top5.set_ylabel("Training Top5")
  ax_top5.legend()
  fig_top5.savefig(os.path.join(save_dir, "top5_convergence.png"))

  ax_eer.set_title("Evaluation EER Convergence")
  ax_eer.set_xlabel("Epochs")
  ax_eer.set_ylabel("EER")
  ax_eer.legend()
  fig_eer.savefig(os.path.join(save_dir, "eer_convergence.png"))
  
  plt.close('all')

if __name__ == "__main__":
  config_file = sys.argv[1] #"xnor_abw_1_wbw_1_20220525-061002"
  with open(config_file, 'r') as f:
    config = json.load(f)
    generate_training_plots(config)
