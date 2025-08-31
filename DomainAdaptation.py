from models.models import Encoder, Classifier, MLPNetworkBig
from utils.utils import select_device, Saver, loadWeights
from utils.data import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
import os

USE_MLP = True

def trainDA(parser, classifier, dataloader, dataloaderTest, encoder, MLP, saver, useMLP):

    learning_rate = parser.lr
    epochs = parser.epochs
    device = select_device(parser.device)

    classifier.to(device)
    encoder.to(device)
    MLP.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.fc.parameters(), lr=learning_rate)
    # Scheduler: Reduce LR by factor of 0.1 every 5 epochs
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    loss_per_epoch = []

    for epoch in range(epochs):
        classifier.train()
        train_loss = 0
        t1 = time.time()

        for idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            latent = encoder(images)

            # TODO : REMEMBER TO UNCOMMENT
            # latent = minMaxNormalization(latent)
            if useMLP :
                latent = MLP(latent)
            outputs = classifier.fc(latent)
            loss = criterion(outputs,labels)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(dataloader)
        loss_per_epoch.append(train_loss)

        # save the best model
        if train_loss < saver.previousBest:
            saver.previousBest = train_loss
            saver.bestEpoch = epoch + 1
            # Save the model weights
            classifierBest = classifier
            saver.saveBest(classifier, "classifier")

        scheduler.step()

        t2 = time.time()

        ValClassAccuracy = Validate(parser, classifier, dataloaderTest)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, val_acc[%]: {ValClassAccuracy:.2f}, Lr: {scheduler.get_last_lr()[0]}, Elapsed time: {(t2-t1):.4f}")

    # Plot Training Curve
    EpochsPlot = [x+1 for x in range(epochs)]
    saver.PlotTrainingCurve(EpochsPlot, loss_per_epoch, "Cross Entropy Loss", parser.domain)

    # Save the last model and store weights to saver
    saver.saveLast(classifier, "classifier")
    saver.classifierWeights = os.path.join(saver.weightsPath, "classifier_best.pth")

    saver.saveInfoTrain()

    return classifierBest

def Validate(parser, classifier, dataloader):

    device = select_device(parser.device)
    classifier.to(device)
    classifier.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = classifier(images)
            _, predicted = torch.max(outputs, 1)

            # Update total metrics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Compute metrics
    total_accuracy = correct / total * 100

    return total_accuracy

def evalClassifier(parser, classifier, dataloader, saver):

    device = select_device(parser.device)
    classifier.to(device)
    classifier.eval()

    correct = 0
    total = 0
    per_class_correct = np.zeros(parser.numClasses)
    per_class_total = np.zeros(parser.numClasses)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = classifier(images)
            _, predicted = torch.max(outputs, 1)

            # Update total metrics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update per-class metrics
            for i in range(parser.numClasses):
                per_class_correct[i] += ((predicted == labels) & (labels == i)).sum().item()
                per_class_total[i] += (labels == i).sum().item()

            # Collect all predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    total_accuracy = correct / total * 100
    per_class_accuracy = (per_class_correct / per_class_total) * 100

    # Print total and per-class accuracy
    print(f"Total Classification Accuracy: {total_accuracy:.2f}%")
    for i, acc in enumerate(per_class_accuracy):
        print(f"Accuracy for class {i}: {acc:.2f}%")

    results = {
        "total_accuracy": total_accuracy,
        "per_class_accuracy": per_class_accuracy}

    saver.saveInfoEval(results, "classification")
    saver.plotConfusionMatrix(all_labels, all_preds, [str(x) for x in range(parser.numClasses)])

def minMaxNormalization(tensor):
    """
    Min-Max normalize a tensor of shape [BatchSize, 1, Features].
    Normalization is done per sample along the last dimension.
    """
    if tensor.dim() == 3 and tensor.shape[1] == 1:
        min_val = tensor.min(dim=2, keepdim=True)[0]
        max_val = tensor.max(dim=2, keepdim=True)[0]
        return (tensor - min_val) / (max_val - min_val + 1e-8)
    
    elif tensor.dim() == 2:
        # Assume shape [BatchSize, Features]
        min_val = tensor.min(dim=1, keepdim=True)[0]
        max_val = tensor.max(dim=1, keepdim=True)[0]
        return (tensor - min_val) / (max_val - min_val + 1e-8)

    elif tensor.dim() == 1:
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val + 1e-8)

    else:
        raise ValueError("Unsupported tensor shape. Expected 1D, 2D, or 3D with shape [B, 1, F]")


##################################################################################################################################

parser = argparse.ArgumentParser("Hyperparameters input")
parser.add_argument("--train_batch", type=int, default=256, help="Batch size for training")
parser.add_argument("--test_batch", type=int, default=32, help="Batch size for testing")
parser.add_argument("--lr", type=float, default=1e-3, help="Training's learning rate")
parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
parser.add_argument("--encoderSource", type=str, default="Results/ImageReconstruction_Source/weights/encoder_best.pth", help="Path to trained encoder")
parser.add_argument("--encoderTarget", type=str, default="Results/ImageReconstruction_Target/weights/encoder_best.pth", help="Path to trained encoder")
parser.add_argument("--MLP", type=str, default="Results/MlpMappingMedoidsNormalized_S2T_Norm_TripletLoss/weights/MLP_best.pth", help="Path to trained encoder")
parser.add_argument("--classifierTarget", type=str, default="Results/ImageClassification_Target/weights/classifier_best.pth", help="Path to trained encoder")
parser.add_argument("--dataTrainTarget", type=str, default="dataset/USPS/TRAIN_SUB", help="relative path of train dataset")
parser.add_argument("--dataTrainSource", type=str, default="dataset/MNIST/TRAIN", help="relative path of train dataset")
parser.add_argument("--dataTest", type=str, default="dataset/USPS/TEST", help="relative path of test dataset")
parser.add_argument("--domain", choices = ["Meds_Norm_Triplet","Meds_Norm_MSE","S2T_Triplet", "S2T_Contrastive", "S2T_MSE", "S2T_CrossDomainContrastiveLoss", "Medoids_S2T_TripletLoss", "Medoids_S2T_MSE", "Medoids_S2T_Contrastive"], help="Domain of the experiment")
parser.add_argument("--device", default="0", help="Device ID to use for training (e.g., '0' for GPU 0, 'cpu' for CPU)")
parser.add_argument("--numClasses", type=int, default=10, help="Number of classes in the dataset")
opt = parser.parse_args()

if __name__ == "__main__" :

    # Parse arguments
    domain = opt.domain
    ScriptName = os.path.basename(__file__).split(".")[0]
    ExperimentPath = os.path.join(os.getcwd(), "Results", f"{ScriptName}_{domain}")
    saver = Saver(ExperimentPath, opt)

    # Load Train dataset and create DataLoader for Source domain
    transform = transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor()])
    datasetDirTrain = os.path.join(os.getcwd(), opt.dataTrainSource)
    DatasetTrain = CustomImageDataset(rootDir=datasetDirTrain,transform=transform)
    DataloaderTrainSource = DataLoader(dataset=DatasetTrain, batch_size=opt.train_batch, shuffle=True, num_workers=4, pin_memory=True)

    # Load model structure
    encoder = Encoder()

    # Load encoder weights and freeze encoder for Source
    encoderBestSource = loadWeights(encoder, opt.encoderSource)
    encoderBestSource.eval()
    for param in encoderBestSource.parameters():
        param.requires_grad = False

    MLP = MLPNetworkBig(128,512,256,128)
    MLPBest = loadWeights(MLP, opt.MLP)
    MLPBest.eval()
    for param in MLPBest.parameters():
        param.requires_grad = False

    # Load encoder weights and freeze encoder for Target
    encoderBestTarget = loadWeights(encoder, opt.encoderTarget)
    encoderBestTarget.eval()
    for param in encoderBestTarget.parameters():
        param.requires_grad = False

    classifier = Classifier(encoderBestTarget, 128, opt.numClasses)
    classifierBestTarget = loadWeights(classifier, opt.classifierTarget)

    # Load Test dataset and create DataLoader
    transform = transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor()])
    datasetDirTest = os.path.join(os.getcwd(), opt.dataTest)
    DatasetTest = CustomImageDataset(rootDir=datasetDirTest,transform=transform)
    DataloaderTest = DataLoader(dataset=DatasetTest, batch_size=opt.test_batch, shuffle=True)


    print("///////// Train Stage 1 ///////////////")
    ClassifierTrained1 = trainDA(opt, classifierBestTarget, DataloaderTrainSource, DataloaderTest,  encoderBestSource, MLPBest, saver, USE_MLP)

    # set false to pass target data
    USE_MLP = False

    # Load Target dataset to retrain Classifier
    transform = transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor()])
    datasetDirTrainTarget = os.path.join(os.getcwd(), opt.dataTrainTarget)
    DatasetTrainTarget = CustomImageDataset(rootDir=datasetDirTrainTarget,transform=transform)
    DataloaderTrainTarget = DataLoader(dataset=DatasetTrainTarget, batch_size=opt.train_batch, shuffle=True, num_workers=4, pin_memory=True)

    # Clean saver saved information related to the previous training
    ExperimentPath2 = os.path.join(os.getcwd(), "Results", f"{ScriptName}_{domain}_finetuned")
    saver2 = Saver(ExperimentPath2, opt)
    print("///////// Train Stage 2 ///////////////")
    # reduce epochs for second training
    # opt.epochs = 25
    ClassifierTrained2 = trainDA(opt, ClassifierTrained1, DataloaderTrainTarget, DataloaderTest, encoderBestTarget, MLPBest, saver2, USE_MLP)



    # Evaluate
    print("///////// Evaluation ///////////////")
    evalClassifier(opt, ClassifierTrained2, DataloaderTest, saver2)
