from models.models import Encoder, Decoder
from utils.utils import select_device, Saver, loadWeights
from utils.data import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import os

def trainAutoencoder(parser, encoder, decoder, dataloader, saver):

    learning_rate = parser.lr
    epochs = parser.epochs
    device = select_device(parser.device)
    encoder.to(device) # transfer encoder to GPU
    decoder.to(device) # transfer decoder to GPU
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        [{'params': encoder.parameters()},
         {'params': decoder.parameters()}], lr=learning_rate)

    loss_per_epoch = []

    # Training Loop
    encoder.train()
    decoder.train()
    for epoch in range(epochs):

        train_loss = 0
        t1 = time.time()
        for idx, (images,_) in enumerate(dataloader):

            images = images.to(device) # use gpu
            # Forward pass
            latent = encoder(images)
            output = decoder(latent)
            loss = criterion(output, images)

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
            saver.saveBest(encoder, "encoder")
            saver.saveBest(decoder, "decoder")

        t2 = time.time()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Elapsed time: {(t2-t1):.4f}")

    # Plot Training Curve
    EpochsPlot = [x+1 for x in range(epochs)]
    saver.PlotTrainingCurve(EpochsPlot, loss_per_epoch, "MSE Loss", parser.domain)

    # Save the last model and store weights to saver
    saver.saveLast(encoder, "encoder")
    saver.saveLast(decoder, "decoder")
    saver.encoderWeights = os.path.join(saver.weightsPath, "encoder_best.pth")
    saver.decoderWeights = os.path.join(saver.weightsPath, "decoder_best.pth")

    saver.saveInfoTrain()

def evalAutoencoder(parser, encoder, decoder, dataloader, saver):
    
    device = select_device(parser.device)
    encoder.to(device)
    decoder.to(device)

    encoder.eval()
    decoder.eval()
    criterion = nn.MSELoss()
    test_loss = 0

    with torch.no_grad():
        for idx, (data,_) in enumerate(dataloader):
            data = data.to(device)
            latent = encoder(data)
            output = decoder(latent)
            loss = criterion(output, data)
            test_loss += loss.item()

    test_loss /= len(dataloader)
    print(f"Test loss = {test_loss:.4f}")

    dataiter = iter(dataloader) # iterate dataloader
    display_images, _ = next(dataiter) # take a batch of images to display
    display_images = display_images.to(device)
    display_outputs = decoder(encoder(display_images))

    saver.saveReconstructions(display_images, display_outputs)
    saver.saveInfoEval(f"Test loss = {test_loss:.4f}", "reconstruction")

##################################################################################################################################

parser = argparse.ArgumentParser("Hyperparameters input")
parser.add_argument("--train_batch", type=int, default=128, help="Batch size for training")
parser.add_argument("--test_batch", type=int, default=32, help="Batch size for testing")
parser.add_argument("--lr", type=float, default=1e-3, help="Training's learning rate")
parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
parser.add_argument("--dataTrain", type=str, default="dataset/MNIST/TRAIN", help="relative path of train dataset")
parser.add_argument("--dataTest", type=str, default="dataset/MNIST/TEST", help="relative path of test dataset")
parser.add_argument("--domain", choices = ["Source", "Target"], help="Domain of the experiment")
parser.add_argument("--device", default="0", help="Device ID to use for training (e.g., '0' for GPU 0, 'cpu' for CPU)")
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
    decoder = Decoder()

    # Train
    print("///////// Training ///////////////")
    trainAutoencoder(opt, encoder, decoder, DataloaderTrain, saver)

    # Load best model weights
    encoderBest = loadWeights(encoder, saver.encoderWeights)
    decoderBest = loadWeights(decoder, saver.decoderWeights)
    
    # Load Test dataset and create DataLoader
    datasetDirTest = os.path.join(os.getcwd(), opt.dataTest)
    DatasetTest = CustomImageDataset(rootDir=datasetDirTest,transform=transform)
    DataloaderTest = DataLoader(dataset=DatasetTest, batch_size=opt.test_batch, shuffle=True)
    
    # Evaluate
    print("///////// Evaluation ///////////////")
    evalAutoencoder(opt, encoderBest, decoderBest, DataloaderTest, saver)