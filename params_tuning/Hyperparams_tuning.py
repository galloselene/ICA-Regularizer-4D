import torch.optim as optim
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import torch
# -----------------------

from model_try01_masked import Abide1DConvNet
from main_train_def import train_val, load_data
k_i =3
# assign correct paths for this fold
paths =[]
paths.append("/data_local/deeplearning/ABIDE_ica/DualRegr/Kfold_" + str(k_i+1) + "/") #timecourse data
paths.append("/data_local/deeplearning/ABIDE_ica/Melodic_Results_v02/Kfold_"+ str(k_i+1) + "/") #ica matrix

# --------------------
def train_and_tune(config):
    perc_data_for_train = 0.33
    train_id_loader, val_loader = load_data(perc_data_for_train, batch_size= config["batch_size"])
    
    model = Abide1DConvNet(n_cluster=25, k_conv1_dim=7)
    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"])
    
    for epoch_i in range(800):
        acc = train_val(model, optimizer, 
            lr_SGD=config["lr_SGD"],
            epoch_i=epoch_i, train_id_loader=train_id_loader, val_loader=val_loader,
            paths=paths)
        
        tune.track.log(mean_accuracy=acc)
        if epoch_i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model.state_dict(), "./model.pth")


search_space = {
    "lr": tune.choice([0.0001, 0.001, 0.01]),
    "batch_size": tune.grid_search([32, 64]),
    "lr_SGD": tune.choice([0.0001, 0.001, 0.01]),
}

ray.init(num_cpus=10, num_gpus=1)
analysis = tune.run(
    train_and_tune,
    num_samples=30,
    scheduler=ASHAScheduler(metric="mean_accuracy", mode="max", grace_period=1),
    config=search_space,
    resources_per_trial={"cpu": 2, "gpu": 1})

dfs = analysis.trial_dataframes
# Plot by epoch
ax = None  # This plots everything on the same plot
for d in dfs.values():
    ax = d.mean_accuracy.plot(ax=ax, legend=False)

print("Best config is", analysis.get_best_config(metric="mean_accuracy"))