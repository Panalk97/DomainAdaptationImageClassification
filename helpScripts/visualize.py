import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import numpy as np

def compute_medoid(tensors):
    """
    Given a list of tensors, return the tensor that has the minimum total distance
    to all other tensors in the list (the medoid).
    """
    if len(tensors) == 0:
        return None  # Edge case: empty class list
    if len(tensors) == 1:
        return tensors[0]  # Only one element; it's the medoid by default

    stacked = torch.stack(tensors)  # Shape: [N, D]
    # Compute pairwise distances using Euclidean norm
    dists = torch.cdist(stacked, stacked, p=2)  # [N, N]
    dist_sums = dists.sum(dim=1)  # [N]
    medoid_idx = torch.argmin(dist_sums)
    return stacked[medoid_idx]

def MedoidList(Latents, classes):
    # Organize  data to classes
    OrganizedData = [[] for x in range(classes)]
    for latent, label in Latents:
        label = label.item()
        OrganizedData[label].append(latent)

    for i in range(len(OrganizedData)):
        OrganizedData[i] = compute_medoid(OrganizedData[i])

    return OrganizedData


# def plot_latents_pca(LatentsTarget, MedoidList, num_classes, title="PCA of Latents"):
#     """
#     Performs PCA on latent vectors and plots them in 2D, coloring each class differently.

#     Args:
#         LatentsTarget: list of (latent_tensor, label_tensor) tuples,
#                        each of shape ([1, 128], [1])
#         num_classes: number of distinct classes
#         title: title for the plot
#     """
#     # Extract and flatten
#     latents = [x[0].squeeze(0).cpu().numpy() for x in LatentsTarget]  # Shape [128]
#     labels = [x[1].item() for x in LatentsTarget]  # Single integer

#     X = np.stack(latents)  # Shape: [N, 128]
#     y = np.array(labels)   # Shape: [N]

#     medLatents = [x.squeeze(0).cpu().numpy() for x in MedoidList]
#     medlabels = [x for x in range(10)]
#     Xmed = np.stack(medLatents)
#     Ymed = np.array(medlabels)

#     # Set up consistent colors
#     colormap = cm.get_cmap("tab10", num_classes)
#     colors = [colormap(i) for i in range(num_classes)]

#     # Apply PCA to
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X)

#     Med_pca = pca.fit_transform(Xmed)

#     # Plot
#     plt.figure(figsize=(10, 7))
#     # For samples
#     for class_id in range(num_classes):
#         idx = y == class_id
#         plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f'Class {class_id}', color=colors[class_id],  alpha=0.6)

#     # For Medoids
#     for idx, med in enumerate(Med_pca):
#         plt.scatter(Med_pca[idx,0], Med_pca[idx,1], marker='X', color=colors[Ymed[idx]], s=300, edgecolors='black', linewidths=1.5, alpha=0.6)


#     plt.xlabel("PCA Component 1")
#     plt.ylabel("PCA Component 2")
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"{title}.png")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA

def plot_latents_pca(LatentsTarget, LatentsMedoids, num_classes, title="PCA of Latents"):
    """
    Performs PCA on latent vectors and plots them in 2D, coloring each class differently.

    Args:
        LatentsTarget: list of (latent_tensor, label_tensor) tuples,
                       each of shape ([1, 128], [1])
        LatentsMedoids: list of (latent_tensor, label_tensor) tuples,
                        each of shape ([1, 128], [1])
        num_classes: number of distinct classes
        title: title for the plot
    """
    # Extract and flatten Target
    latents = [x[0].squeeze(0).cpu().numpy() for x in LatentsTarget]  # Shape [128]
    labels = [x[1].item() for x in LatentsTarget]  # Single integer

    X = np.stack(latents)  # Shape: [N, 128]
    y = np.array(labels)   # Shape: [N]

    # Extract and flatten Medoids
    medLatents = [x[0].squeeze(0).cpu().numpy() for x in LatentsMedoids]
    medLabels = [x[1].item() for x in LatentsMedoids]

    Xmed = np.stack(medLatents)
    Ymed = np.array(medLabels)

    # Set up consistent colors
    colormap = cm.get_cmap("tab10", num_classes)
    colors = [colormap(i) for i in range(num_classes)]

    # Concatenate the two sets
    X_combined = np.concatenate([X, Xmed], axis=0)  # shape: [N + M, 128]

    # Fit PCA on the combined set
    pca = PCA(n_components=2)
    X_combined_pca = pca.fit_transform(X_combined)  # shape: [N + M, 2]

    # Split back into target and medoid projections
    X_pca = X_combined_pca[:len(X)]        # First N points: target latents
    Med_pca = X_combined_pca[len(X):]     # Remaining M points: medoid latents

    # Plot
    plt.figure(figsize=(10, 7))
    # For samples
    for class_id in range(num_classes):
        idx = y == class_id
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f'Class {class_id}', color=colors[class_id], alpha=0.6)

    # For Medoids
    for class_id in range(num_classes):
        idx = Ymed == class_id
        plt.scatter(Med_pca[idx, 0], Med_pca[idx, 1], marker='*', color=colors[class_id], s=200,
                    edgecolors='black', linewidths=1., alpha=0.8)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.close()


CLASSES = 10  # Adjust this based on your dataset
# Load latents
LatentsPathTarget = "/home/peftis/pantelis/General/MSc Thesis/Domain Adaptation/code/RefactoredCode/Results/saveLatents_Target/Latents_Target.pt"
LatentsPathSource = "/home/peftis/pantelis/General/MSc Thesis/Domain Adaptation/code/RefactoredCode/Results/saveLatents_S2T_Triplet_Medoids2/Latents_S2T_Triplet_Medoids2.pt"

LatentsTarget = torch.load(LatentsPathTarget)
LatentsSource = torch.load(LatentsPathSource)


# MedsTarget = MedoidList(LatentsTarget, CLASSES)
plot_latents_pca(LatentsTarget, LatentsSource, num_classes=CLASSES, title="Mapped Source (Triplet & Medoids) and Target Latent Representations PCA")
# plot_latents_pca(LatentsSource, MedsTarget, num_classes=CLASSES, title="PCA_CrossDomainContrastiveLoss")