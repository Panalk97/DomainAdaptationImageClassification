import torch
import os
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import json

def select_device(option=None):
    if option == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            device = torch.device(f'cuda:{option}')
        except:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    return device

def createDirectory(directory):
    os.makedirs(directory, exist_ok=True)

def batchOrganizer(data, batchSize):

    # Organize source data into batches
    dataBatched = []
    total_samples = len(data)

    # Extract features and labels separately
    features = torch.stack([item[0] for item in data])  # Shape: (N, 128)
    labels = torch.stack([item[1] for item in data])  # Shape: (N,)

    # Create batches
    for i in range(0, total_samples - total_samples % batchSize, batchSize):
        batch_features = features[i:i+batchSize]  # Shape: (16, 128)
        batch_labels = labels[i:i+batchSize]  # Shape: (16,)
        dataBatched.append((batch_features, batch_labels))

    # commented only for the case of CrossDomainContrastiveLoss
    # # Handle the remaining samples if any
    if total_samples % batchSize != 0:
        print("Adding the last smaller batch")
        batch_features = features[i+batchSize:]  # Shape: (remaining, 128)
        batch_labels = labels[i+batchSize:]  # Shape: (remaining,)
        dataBatched.append((batch_features, batch_labels))
    # commented only for the case of CrossDomainContrastiveLoss
    # Example of accessing a batch
    print(f"Total batches: {len(dataBatched)}")
    print(f"First batch shape: Features {dataBatched[0][0].shape}, Labels {dataBatched[0][1].shape}")

    return dataBatched

def batchOrganizerCrossDomainContrastive(data, batchSize):

    # Organize source data into batches
    dataBatched = []
    total_samples = len(data)

    # Extract features and labels separately
    features = torch.stack([item[0] for item in data])  # Shape: (N, 128)
    labels = torch.stack([item[1] for item in data])  # Shape: (N,)

    # Create batches
    for i in range(0, total_samples - total_samples % batchSize, batchSize):
        batch_features = features[i:i+batchSize]  # Shape: (16, 128)
        batch_labels = labels[i:i+batchSize]  # Shape: (16,)
        dataBatched.append((batch_features, batch_labels))

    # commented only for the case of CrossDomainContrastiveLoss
    # # Handle the remaining samples if any
    # if total_samples % batchSize != 0:
    #     print("Adding the last smaller batch")
    #     batch_features = features[i+batchSize:]  # Shape: (remaining, 128)
    #     batch_labels = labels[i+batchSize:]  # Shape: (remaining,)
    #     dataBatched.append((batch_features, batch_labels))
    # commented only for the case of CrossDomainContrastiveLoss
    # Example of accessing a batch
    print(f"Total batches: {len(dataBatched)}")
    print(f"First batch shape: Features {dataBatched[0][0].shape}, Labels {dataBatched[0][1].shape}")

    return dataBatched

def loadWeights(model, weightsPath):

    try:
        model.load_state_dict(torch.load(weightsPath))
        print(f"Weights loaded from {weightsPath}")
        return model
    except FileNotFoundError:
        print(f"File not found: {weightsPath}")
        raise

class Saver:
    '''
    This is a class to save model weights in a best/last logic
    depending on the training loss. Other results
    related to the training and evaluation will also be saved.
    '''
    def __init__(self, ExpPath, parser):
        self.ExpPath = ExpPath
        self.previousBest = np.inf
        self.bestEpoch = -1
        self.parser = parser
        self.weightsPath = os.path.join(self.ExpPath, "weights")
        self.metricsPath = os.path.join(self.ExpPath, "metrics")
        self.encoderWeights = None
        self.decoderWeights = None
        self.classifierWeights = None
        self.MLPWeights = None
        self.createDirectories()

    def saveBest(self, model, modelname):
        torch.save(model.state_dict(), os.path.join(self.weightsPath, f"{modelname}_best.pth"))

    def saveLast(self, model, modelname):
        torch.save(model.state_dict(), os.path.join(self.weightsPath, f"{modelname}_last.pth"))

    def createDirectories(self):
            os.makedirs(self.ExpPath, exist_ok=True)
            os.makedirs(self.weightsPath, exist_ok=True)
            os.makedirs(self.metricsPath, exist_ok=True)

    def PlotTrainingCurve(self, X, Y, ylabel, domain):
        plt.figure()
        plt.plot(X, Y, marker='.')
        plt.xlabel("Epochs")
        plt.ylabel(ylabel) # loss
        plt.title(f"Training Loss Curve - {domain}")
        plt.grid()
        plt.savefig(os.path.join(self.metricsPath,"Training_loss_curve.jpg"))
        # plt.show()

    def plotConfusionMatrix(self, groundTruth, prediction, classNames):
        confMatrix = confusion_matrix(groundTruth,prediction)
        plt.figure(figsize=(10, 8))
        plt.imshow(confMatrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)

        # Display the numbers inside the confusion matrix
        for i, j in np.ndindex(confMatrix.shape):
            plt.text(j, i, format(confMatrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if confMatrix[i, j] > confMatrix.max() / 2 else "black")

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(self.metricsPath,"ConfusionMatrix.jpg"))

    def saveReconstructions(self, display_images, display_outputs):
        self.evalVisualizer(torchvision.utils.make_grid(display_images.cpu()), os.path.join(self.metricsPath, "original_input_batch.jpg"), "Input Images")
        self.evalVisualizer(torchvision.utils.make_grid(display_outputs.cpu()),  os.path.join(self.metricsPath, "predicted_input_batch.jpg"), "Reconstructed Images")

    def saveInfoTrain(self):
        data = {"epochs": self.parser.epochs,
                'best_epoch': self.bestEpoch,
                "batch_size": self.parser.train_batch,
                "learning_rate": self.parser.lr,
                "domain": self.parser.domain}
        with open(os.path.join(self.ExpPath, "train_info.json"), "w") as f:
            json.dump(data, f, indent=4)
        f.close()

    def saveInfoEval(self, results, problem):

        with open(os.path.join(self.ExpPath, "evaluation_info.txt"), "w") as f:
            if problem == "classification":
                f.write("Classification Evaluation Results:\n")
                f.write(f"Total Accuracy: {results['total_accuracy']:.2f}\n")
                for i, acc in enumerate(results["per_class_accuracy"]):
                    f.write(f"Accuracy for class {i}: {acc:.2f}%\n")
            elif problem == "reconstruction":
                f.write("Reconstruction Evaluation Results:\n")
                f.write(results)
                f.write("\n")

        f.close()

    def evalVisualizer(self, image, savePath, title):
        npImage = image.numpy()
        plt.figure()
        plt.imshow(np.transpose(npImage,(1,2,0)))
        plt.title(title)
        plt.savefig(savePath)