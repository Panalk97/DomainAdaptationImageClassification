from models.models import Encoder, Classifier
from utils.utils import select_device, Saver, loadWeights
from utils.data import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
import os

def trainClassifier(parser, classifier, dataloader, saver):

    learning_rate = parser.lr
    epochs = parser.epochs
    device = select_device(parser.device)

    classifier.to(device) # transfer encoder to GPU

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.fc.parameters(), lr=learning_rate)

    loss_per_epoch = []

    # Training Loop
    classifier.train()
    for epoch in range(epochs):

        train_loss = 0
        t1 = time.time()
        for idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = classifier(images)
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
            saver.saveBest(classifier, "classifier")

        t2 = time.time()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Elapsed time: {(t2-t1):.4f}")

    # Plot Training Curve
    EpochsPlot = [x+1 for x in range(epochs)]
    saver.PlotTrainingCurve(EpochsPlot, loss_per_epoch, "Cross Entropy Loss", parser.domain)

    # Save the last model and store weights to saver
    saver.saveLast(classifier, "classifier")
    saver.classifierWeights = os.path.join(saver.weightsPath, "classifier_best.pth")
    
    saver.saveInfoTrain()

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

##################################################################################################################################

parser = argparse.ArgumentParser("Hyperparameters input")
parser.add_argument("--train_batch", type=int, default=128, help="Batch size for training")
parser.add_argument("--test_batch", type=int, default=32, help="Batch size for testing")
parser.add_argument("--lr", type=float, default=1e-3, help="Training's learning rate")
parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
parser.add_argument("--encoder", type=str, default="Results/ImageReconstruction_Source/weights/encoder_best.pth", help="Path to trained encoder")
parser.add_argument("--dataTrain", type=str, default="dataset/MNIST/TRAIN", help="relative path of train dataset")
parser.add_argument("--dataTest", type=str, default="dataset/MNIST/TEST", help="relative path of test dataset")
parser.add_argument("--domain", choices = ["Source", "Target"], help="Domain of the experiment")
parser.add_argument("--device", default="0", help="Device ID to use for training (e.g., '0' for GPU 0, 'cpu' for CPU)")
parser.add_argument("--numClasses", type=int, default=10, help="Number of classes in the dataset")
opt = parser.parse_args()

if __name__ == "__main__":

    # Parse arguments
    domain = opt.domain
    ScriptName = os.path.basename(__file__).split(".")[0]
    ExperimentPath = os.path.join(os.getcwd(), "Results", f"{ScriptName}_{domain}")
    saver = Saver(ExperimentPath, opt)

    # Load Train dataset and create DataLoader
    transform = transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor()])
    datasetDirTrain = os.path.join(os.getcwd(), opt.dataTrain)
    DatasetTrain = CustomImageDataset(rootDir=datasetDirTrain,transform=transform)
    DataloaderTrain = DataLoader(dataset=DatasetTrain, batch_size=opt.train_batch, shuffle=True, num_workers=4, pin_memory=True)

    # Load model structure
    encoder = Encoder()

    # Load encoder weights and freeze encoder
    encoderBest = loadWeights(encoder, opt.encoder)
    encoderBest.eval()
    for param in encoderBest.parameters():
        param.requires_grad = False

    # Prepare classifier
    classifier = Classifier(encoderBest, 128, 10)

    # Train classifier
    print("///////// Training ///////////////")
    trainClassifier(opt, classifier, DataloaderTrain, saver)


    # Load best model weights
    classifierBest = loadWeights(classifier, saver.classifierWeights)

    # Load Test dataset and create DataLoader
    datasetDirTest = os.path.join(os.getcwd(), opt.dataTest)
    DatasetTest = CustomImageDataset(rootDir=datasetDirTest,transform=transform)
    DataloaderTest = DataLoader(dataset=DatasetTest, batch_size=opt.test_batch, shuffle=True)

    # Evaluate
    print("///////// Evaluation ///////////////")
    evalClassifier(opt, classifierBest, DataloaderTest, saver)