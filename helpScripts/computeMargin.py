'''
This is a script to calculate the margin to be used in the computation of
Contrastive loss. The margin is calculated as the minimum distance between
classes in the target feature space.
'''
import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

def calculateMargin(Centroids, classes):

    Distances = np.zeros((classes, classes))

    for i in range(classes):
        for j in range(classes):
            if i != j:
                Distances[i][j] = torch.dist(Centroids[i], Centroids[j]).item()


    # Pretty print full distance matrix
    distance_df = pd.DataFrame(Distances, columns=[f"Class {i}" for i in range(classes)],
                                        index=[f"Class {i}" for i in range(classes)])
    print("\n📏 Pairwise Distance Matrix:\n")
    print(distance_df.round(2))

    # Calculate the margin as the minimum distance between classes
    keepi = -1
    keepj = -1
    min_distance = np.inf
    for i in range(classes):
        for j in range(classes):
            if i != j and Distances[i][j] < min_distance:
                min_distance = min(min_distance, Distances[i][j])
                keepi = i
                keepj = j

    print(f"\nMinimum distance between classes {keepi} and {keepj}: {min_distance:.2f}")

def plotCentroidLatentsPCA(Centroids, domain):
    # Convert to numpy
    data = np.stack([c.cpu().numpy().flatten() for c in Centroids])

    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    # Define distinct colors for each class
    colors = plt.cm.tab10.colors

    # === PCA Plot ===
    plt.figure(figsize=(6, 5))
    for i in range(len(pca_result)):
        plt.scatter(pca_result[i, 0], pca_result[i, 1], color=colors[i], label=f'Class {i}')
    plt.title(f'PCA of Centroids - {domain}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='upper right', fontsize='small', frameon=True)
    plt.tight_layout()
    plt.savefig("pca_centroids.png", dpi=300)
    # plt.show()

def plotCentroidLatentsTSNE(Centroids, domain):
    # Convert to numpy
    data = np.stack([c.cpu().numpy().flatten() for c in Centroids])

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_result = tsne.fit_transform(data)

    # Define distinct colors for each class
    colors = plt.cm.tab10.colors

    # === t-SNE Plot ===
    plt.figure(figsize=(6, 5))
    for i in range(len(tsne_result)):
        plt.scatter(tsne_result[i, 0], tsne_result[i, 1], color=colors[i], label=f'Class {i}')
    plt.title(f't-SNE of Centroids - {domain}')
    plt.xlabel('t-SNE1')
    plt.ylabel('t-SNE2')
    plt.legend(loc='upper right', fontsize='small', frameon=True)
    plt.tight_layout()
    plt.savefig("tsne_centroids.png", dpi=300)
    # plt.show()


# TARGETLATENTSPATH = "/home/peftis/pantelis/General/MSc Thesis/Domain Adaptation/code/RefactoredCode/Results/saveLatents_Target/Latents_Target.pt"
TARGETLATENTSPATH = "/home/peftis/pantelis/General/MSc Thesis/Domain Adaptation/code/RefactoredCode/Results/saveLatents_Source/Latents_Source.pt"
CLASSES = 10

if __name__ == "__main__":

    LatentsTarget = torch.load(TARGETLATENTSPATH)

    # organize target data to classes
    OrganizedLatentsTarget = [[] for x in range(CLASSES)]
    Centroids = OrganizedLatentsTarget
    for latent, label in LatentsTarget:
        label = label.item()
        OrganizedLatentsTarget[label].append(latent)

    for i in range(len(OrganizedLatentsTarget)):
        Centroids[i] = torch.mean(torch.stack(OrganizedLatentsTarget[i]), dim=0)

    # Calculate the margin
    calculateMargin(Centroids, CLASSES)

    # Plot the centroids using PCA
    # plotCentroidLatentsPCA(Centroids, "Target")
    plotCentroidLatentsPCA(Centroids, "Source")


    # Plot the centroids using t-SNE
    # plotCentroidLatentsTSNE(Centroids, "Target")
    plotCentroidLatentsTSNE(Centroids, "Source")
