import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss functions for meta-learning
def meta_loss_mse(logits, targets):
    """
    MSE
    """
    probs = F.softmax(logits, dim=1)
    one_hot_targets = F.one_hot(targets, num_classes=probs.shape[1]).float()
    return F.mse_loss(probs, one_hot_targets)

def meta_loss_cross_entropy(logits, targets):
    """
    Cross-entropy
    """
    return F.cross_entropy(logits, targets)

# Regularization
def regularizer_class_dist(generator, x, y, reg_lambda=0.05):
    """
    Average intra-class distance
    """
    num_classes = len(torch.unique(y))
    x_flat = x.view(x.size(0), -1)
    dim = x_flat.shape[1]
    
    x_flat = x_flat / (x_flat.norm(dim=1, keepdim=True) + 1e-8)
    
    diversity = 0
    valid_classes = 0
    
    for c in range(num_classes):
        class_mask = (y == c)
        if class_mask.sum() < 2:
            continue
            
        class_x = x_flat[class_mask]
        n_c = class_x.shape[0]
        
        dists = torch.cdist(class_x, class_x, p=2)
        
        mask = ~torch.eye(n_c, dtype=bool, device=x.device)
        intra_class_dist = dists[mask].mean()
        
        intra_class_dist = intra_class_dist / 2.0
        
        diversity += intra_class_dist
        valid_classes += 1
    
    if valid_classes > 0:
        diversity = diversity / valid_classes
    
    return -reg_lambda * diversity

def regularizer_svd(generator, x, y, reg_lambda=1, threshold=1):
    """
    Penalties for singular values below the threshold
    """
    reg_loss = 0.0
    count = 0

    for module in generator.modules():
        if isinstance(module, nn.Linear):
            try:
                U, S, V = torch.svd(module.weight)
                small = torch.clamp(threshold - S, min=0)
                reg_loss += torch.sum(small ** 2) / len(S)
                count += 1
            except:
                pass
    
    if count == 0:
        if hasattr(generator, 'data'):
            data = generator.data
            data_flat = data.view(data.size(0), -1)
            try:
                U, S, V = torch.svd(data_flat)
                small = torch.clamp(threshold - S, min=0)
                reg_loss += torch.sum(small ** 2) / len(S)
                count += 1
            except:
                pass
    
    if count > 0:
        reg_loss /= count
    
    return reg_lambda * reg_loss