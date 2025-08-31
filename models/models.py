import torch
import torch.nn as nn
import torch.nn.init as init

# Define Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [1, 28, 28] -> [32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [32, 14, 14] -> [64, 7, 7]
            nn.ReLU(),
            nn.Flatten(),  # [64, 7, 7] -> [3136]
            nn.Linear(64 * 7 * 7, 128),  # Latent space
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

# Define Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(128, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [64, 7, 7] -> [32, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [32, 14, 14] -> [1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

class Classifier(nn.Module):
    def __init__(self, encoder, latentDim, numClasses):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(latentDim, 256),  # First FC layer
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout to prevent overfitting
            nn.Linear(256, 128),  # Second FC layer
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout to prevent overfitting
            nn.Linear(128, numClasses)  # Final output layer
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(latentDim, 128),  # First FC layer
        #     nn.ReLU(),
        #     nn.Dropout(0.3),  # Dropout to prevent overfitting
        #     nn.Linear(128, 64),  # Second FC layer
        #     nn.ReLU(),
        #     nn.Dropout(0.3),  # Dropout to prevent overfitting
        #     nn.Linear(64, numClasses)  # Final output layer
        # )

    def forward(self,x):
        with torch.no_grad(): #ensure encoder weigths are frozen
            x = self.encoder(x)
        x = self.fc(x)
        return x

class MLPNetworkBig(nn.Module):
    def __init__(self, inputVec, inputlayer, hiddenlayer, outputlayer):
        super(MLPNetworkBig, self).__init__()
        self.dropout = nn.Dropout(0.4)

        self.layer1 = nn.Linear(in_features=inputVec, out_features=inputlayer)
        self.layer2 = nn.Linear(in_features=inputlayer, out_features=hiddenlayer)
        self.layer3 = nn.Linear(in_features=hiddenlayer, out_features=outputlayer)

        self.initialize_weights()  # Apply Kaiming Initialization

    def initialize_weights(self):
        """Apply Kaiming Initialization to layers."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')  # He Normal
                if layer.bias is not None:
                    init.zeros_(layer.bias)  # Bias set to zero

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)  # No activation on output (assumes loss function handles it)

        return x
