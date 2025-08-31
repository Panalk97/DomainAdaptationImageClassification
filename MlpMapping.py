from models.models import MLPNetworkBig
from utils.utils import select_device, Saver, loadWeights, batchOrganizer
from utils.data import CustomImageDataset
from utils.losses import CustomMSE, ContrastiveLoss, CustomTripletLoss, ContrastiveLoss2, UnifiedTripletAlignmentLoss
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
import os

def trainMLP(parser, latentSource, latentTarget, MLP,  saver):

    '''
    mapps data from the second argument (latentSource) to the third argument (latentTarget)
    so third argument is a list of oganized data by class
    '''

    learning_rate = parser.lr
    epochs = parser.epochs
    device = select_device(parser.device)

    MLP.to(device) # transfer MLP to GPU
    if parser.loss == "MSE":
        criterion = CustomMSE(parser.centroid)
        plotTitle = "Custom MSE Loss"
    elif parser.loss == "ContrastiveLoss": # remember to uncomment 
        # criterion = ContrastiveLoss(parser.margin) 
        criterion = ContrastiveLoss2(parser.margin) 
        plotTitle = "Contrastive Loss"
    elif parser.loss == "TripletLoss":
        criterion = CustomTripletLoss()
        plotTitle = "Triplet Loss"
    elif parser.loss == "UnifiedTripletAlignmentLoss":
        criterion = UnifiedTripletAlignmentLoss()
        plotTitle = "UnifiedTripletAlignment Loss"


    optimizer = optim.Adam(MLP.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(MLP.parameters(), lr=learning_rate)


    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=parser.stepSize, gamma=parser.gamma)
    loss_per_epoch = [] # for plots

    for epoch in range(epochs):
        MLP.train()
        train_loss = 0
        t1 = time.time()

        # loop to iterate data from Source domain to be mapped to data from target domain
        # import pdb;pdb.set_trace()
        for idxS, (dataS,labelsS) in enumerate(latentSource):

            # predict
            predict = MLP(dataS)

            if parser.loss == "ContrastiveLoss":
                loss = criterion.forwardPairs(predict, labelsS, latentTarget)
            elif parser.loss == "TripletLoss":
                loss = criterion.forwardTriplets(predict, labelsS, latentTarget)
            elif parser.loss == "MSE":
                loss = criterion(predict, labelsS, latentTarget)
            elif parser.loss == "UnifiedTripletAlignmentLoss":
                loss = criterion(predict, labelsS, latentTarget)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Reduce learning rate after each epoch
        # scheduler.step()
        train_loss /= len(latentSource)
        loss_per_epoch.append(train_loss)

        # save the best model
        if train_loss < saver.previousBest:
            saver.previousBest = train_loss
            saver.bestEpoch = epoch + 1
            # Save the model weights
            saver.saveBest(MLP, "MLP")

        t2 = time.time()

        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.6f}, Elapsed time: {(t2-t1):.4f} Learning rate: {scheduler.get_last_lr()}")
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.6f}, Elapsed time: {(t2-t1):.4f}")

    # Plot Training Curve
    EpochsPlot = [x+1 for x in range(epochs)]
    saver.PlotTrainingCurve(EpochsPlot, loss_per_epoch, plotTitle, parser.domain)

     # Save the last model and store weights to saver
    saver.saveLast(MLP, "MLP")
    saver.saveInfoTrain()
    
##################################################################################################################################

parser = argparse.ArgumentParser("Hyperparameters input")
parser.add_argument("--lr", type=float, default=1e-4, help="Training's learning rate")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs") #should go to 25 for time consuming trainings 
parser.add_argument("--train_batch", type=int, default=16, help="Batch size")
parser.add_argument("--SourceLatents", type=str, default="Results/saveLatents_Source/Latents_Source.pt", help="Source's latent representations")
parser.add_argument("--TargetLatents", type=str, default="Results/saveLatents_Target/Latents_Target.pt", help="Target's latent representations")
parser.add_argument("--loss", type=str, choices=["MSE", "ContrastiveLoss", "TripletLoss", "UnifiedTripletAlignmentLoss"], default="MSE", help="Criterion to use for training")
parser.add_argument("--classes", type=int, default=10, help="number of classes")
parser.add_argument("--domain", type=str, default="S2T", help="Domain translation")
parser.add_argument("--device", type=int, default=0, help="Device to use")
parser.add_argument("--centroid", type=bool, default=True, help="Use centroid of target classes")
parser.add_argument("--margin", type=float, default=1.0, help="Margin for contrastive loss") # margin for contrastive loss 15.3 
opt = parser.parse_args()

if __name__ == "__main__":

   # Parse arguments
    domain = opt.domain
    ScriptName = os.path.basename(__file__).split(".")[0]
    ExperimentPath = os.path.join(os.getcwd(), "Results", f"{ScriptName}_{domain}_{opt.loss}")
    saver = Saver(ExperimentPath, opt)

    # Load data
    LatentsSource = torch.load(opt.SourceLatents)
    LatentsTarget = torch.load(opt.TargetLatents)

    # Load model structure
    MLP = MLPNetworkBig(128,512,256,128)

    # Load Source data 
    sourceLatentsBatched = batchOrganizer(LatentsSource, opt.train_batch)

    # Organize target data to classes
    OrganizedLatentsTarget = [[] for x in range(opt.classes)]
    for latent, label in LatentsTarget:
        label = label.item()
        OrganizedLatentsTarget[label].append(latent)

    if opt.centroid:
        # Calculate the centroid of each class
        for i in range(len(OrganizedLatentsTarget)):
            OrganizedLatentsTarget[i] = torch.mean(torch.stack(OrganizedLatentsTarget[i]), dim=0)

        print(len(OrganizedLatentsTarget))
        print(OrganizedLatentsTarget[0].shape)

    print("///////// Training MLP ///////////////")
    trainMLP(opt, sourceLatentsBatched, OrganizedLatentsTarget, MLP,  saver)
