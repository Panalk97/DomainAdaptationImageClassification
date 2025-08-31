import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TripletMarginLoss

class CustomMSE(nn.Module):

    def __init__(self, centroid=False):
        super(CustomMSE, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')  # No reduction for individual loss computation
        self.centroid = centroid

    def forward(self, inputs, labels, targets):
        batch_loss = 0

        for sx in range(len(inputs)):
            index = labels[sx].item()  # Class label of the source vector
            target_vectors = targets[index]  # Target vectors of the same class
            # if len(target_vectors) == 0:
            #     continue  # Skip if no target vectors for the class

            # Convert target_vectors to a tensor and move it to the same device as inputs
            if not(self.centroid):
                target_vectors = torch.stack(target_vectors).to(inputs.device)  # Shape: [N, 128]

            # Compute the pairwise MSE
            diffs = target_vectors - inputs[sx].unsqueeze(0)  # Shape: [N, 128]
            squared_diffs = diffs.pow(2)  # Element-wise square
            losses = squared_diffs.mean(dim=1)  # MSE for each target vector
            class_loss = losses.mean()  # Average loss for the class

            batch_loss += class_loss

        return batch_loss / len(inputs) #if len(inputs) > 0 else 0

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss Function for Siamese Networks.

    Args:
        margin (float): Minimum distance at which dissimilar pairs are considered sufficiently apart.

    If the pair is similar (y = 0), the loss is calculated as the squared distance between the embeddings.
    If the pair is dissimilar (y = 1), the loss is calculated as the squared margin minus the distance (if positive), squared.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, y):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss = torch.mean(
            (1 - y) * torch.pow(euclidean_distance, 2) +
            y * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss

    def forwardPairs(self, inputs, labels, target):
        total_loss = 0.0
        num_pairs = 0

        for sx in range(len(inputs)):
            index = labels[sx].item()
            input_embed = inputs[sx].unsqueeze(0)  # Ensure correct shape

            for tx in range(len(target)):
                target_embed = target[tx].unsqueeze(0)  # Ensure correct shape
                y = torch.tensor(0.0) if index == tx else torch.tensor(1.0)

                loss = self.forward(input_embed, target_embed, y)
                total_loss += loss
                num_pairs += 1

        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)


class ContrastiveLoss2(nn.Module):
    '''
    This loss computes the distance for 2 pairs for each sample:
    - Positive pair: (anchor, positive) where both belong to the same class
    - Negative pair: (anchor, negative) where both belong to different classes , as negative it uses t
    the anchor of a different class that is closer to the sample anchor
    '''
    def __init__(self, margin=1.0):
        super(ContrastiveLoss2, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, y):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss = torch.mean(
            (1 - y) * torch.pow(euclidean_distance, 2) +
            y * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss

    def forwardPairs(self, inputs, labels, target):
        """
        Args:
            inputs: Tensor of shape [N, D] - anchor samples
            labels: Tensor of shape [N] - class labels for each anchor
            target: List of class embeddings, where target[i] is the prototype of class i (1D tensor of shape [D])
        """
        total_loss = 0.0
        num_pairs = 0
        device = inputs.device

        # Stack target list into a tensor of shape [num_classes, D]
        target_tensor = torch.stack(target).to(device)  # shape [C, D]
        num_classes = target_tensor.size(0)

        for i in range(len(inputs)):
            anchor = inputs[i].unsqueeze(0).to(device)  # shape [1, D]
            label = labels[i].item()

            # Positive pair: anchor vs same-class target
            positive = target_tensor[label].unsqueeze(0)  # shape [1, D]
            loss_pos = self.forward(anchor, positive, torch.tensor(0.0, device=device))
            total_loss += loss_pos
            num_pairs += 1

            # Create mask to exclude current label
            mask = torch.ones(num_classes, dtype=torch.bool, device=device)
            mask[label] = False
            negative_targets = target_tensor[mask]  # shape [C-1, D]

            # Find closest negative target
            anchor_expanded = anchor.expand_as(negative_targets)  # shape [C-1, D]
            distances = F.pairwise_distance(anchor_expanded, negative_targets)  # shape [C-1]
            min_idx = torch.argmin(distances).item()
            closest_negative = negative_targets[min_idx].unsqueeze(0)  # shape [1, D]

            # Negative pair
            loss_neg = self.forward(anchor, closest_negative, torch.tensor(1.0, device=device))
            total_loss += loss_neg
            num_pairs += 1

        if num_pairs == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / num_pairs


class CustomTripletLoss(nn.Module):
    """
    Custom Triplet Loss builder that creates triplets based on:
    - Anchor: input sample
    - Positive: target sample of the same class
    - Negative: target sample of different class and minimum distance from anchor
    """
    def __init__(self, margin=1.0, p=2):
        super(CustomTripletLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.triplet_loss = TripletMarginLoss(margin=margin, p=p)

    def forwardTriplets(self, inputs, labels, target):
        total_loss = 0.0
        valid_triplets = 0

        for sx in range(len(inputs)):
            anchor = inputs[sx].unsqueeze(0)  # shape: (1, D)
            anchor_label = labels[sx].item()

            # --------- Find Positive ---------
            positive = target[anchor_label].unsqueeze(0)

            # --------- Find Negative ---------
            min_dist = float('inf')
            negative = None
            for tx in range(len(target)):
                if tx == anchor_label:
                    continue  # skip same class

                cand = target[tx].unsqueeze(0)
                dist = F.pairwise_distance(anchor, cand, p=self.p)
                if dist.item() < min_dist:
                    min_dist = dist.item()
                    negative = cand

            if negative is None:
                continue  # skip if no negative found

            # --------- Compute Triplet Loss ---------
            loss = self.triplet_loss(anchor, positive, negative)
            total_loss += loss
            valid_triplets += 1

        return total_loss / valid_triplets if valid_triplets > 0 else torch.tensor(0.0)

class CrossDomainContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.35):
        """
        Cross-Domain Contrastive Loss for comparing source and target domain features.

        Args:
            temperature (float): Temperature scaling parameter for contrastive softmax.
        """
        super(CrossDomainContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, f_s, y_s, f_t, y_t):
        """
        Args:
            f_s (Tensor): Source features of shape (N, D)
            y_s (Tensor): Source labels of shape (N,)
            f_t (Tensor): Target features of shape (M, D)
            y_t (Tensor): Target labels of shape (M,)

        Returns:
            Tensor: Scalar loss value
        """
        # Normalize embeddings
        f_s = F.normalize(f_s, dim=1)  # (N, D)
        f_t = F.normalize(f_t, dim=1)  # (M, D)
        # remove extra dimension
        f_s = torch.squeeze(f_s, dim=1)
        # f_t = torch.squeeze(f_s, dim=1) #TODO it was wrong
        f_t = torch.squeeze(f_t, dim=1)
        # import pdb;pdb.set_trace()
        # Compute cosine similarity matrix: (N, M)
        sim_matrix = torch.matmul(f_s, f_t.T) / self.temperature

        # Create mask for positive pairs: (N, M)
        pos_mask = y_s.unsqueeze(1) == y_t.unsqueeze(0)

        # For numerical stability
        logits = sim_matrix
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Apply softmax over target samples for each source sample
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)  # (N, M)

        # Compute mean log-prob over positive pairs
        mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-9)

        # Final loss: negative of the mean over all source samples with at least one positive
        # import pdb;pdb.set_trace()
        valid_mask = pos_mask.sum(dim=1) > 0
        valid_mask = valid_mask.squeeze(1)  # Now shape is [64]
        loss = -mean_log_prob_pos[valid_mask].mean()

        return loss