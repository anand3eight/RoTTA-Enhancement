import torchattacks

# Continuous Transformation Attack implementation
def continuous_transformation_attack(model, images, labels, device='cuda'):

    # Define the attacks
    pgd_100 = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=100)
    cw_30 = torchattacks.CW(model, c=1e-4, kappa=0, steps=30)
    pgd_50 = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=50)
    cw_50 = torchattacks.CW(model, c=1e-4, kappa=0, steps=50)
    pgd_20 = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20)

    # Sequence of attacks
    attack_sequence = [
        (pgd_100, 100), 
        (cw_30, 30), 
        (pgd_50, 50), 
        (cw_50, 50), 
        (pgd_20, 20), 
        (pgd_50, 50), 
        (cw_50, 50), 
        (pgd_50, 50), 
        (pgd_100, 100)
    ]
        
    for attack, _ in attack_sequence:
        images = attack(images, labels)  # Apply the current attack
            
    return images

# Mixed Batch Attack implementation
def mixed_batch_attack(model, images, labels, device='cuda'):

    # Define the attacks
    pgd_20 = torchattacks.PGD(model, eps=0.3, alpha=2/255, steps=20)
    cw_attack = torchattacks.CW(model, c=1e-4, kappa=0, steps=30)
    batch_size = images.size(0)

    # Indices for different attack groups
    indices = torch.randperm(batch_size)
    pgd_indices = indices[:int(0.3 * batch_size)]
    cw_indices = indices[int(0.3 * batch_size):int(0.6 * batch_size)]
    clean_indices = indices[int(0.6 * batch_size):]

    attacked_images = images.clone()
    attacked_images[pgd_indices] = pgd_20(images[pgd_indices], labels[pgd_indices])
    attacked_images[cw_indices] = cw_attack(images[cw_indices], labels[cw_indices])

    return attacked_images
