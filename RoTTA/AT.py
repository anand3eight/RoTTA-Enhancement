import torch
import torch.nn.functional as F
import torch.nn as nn

def fgsm_kl(model, 
            teacher_model, 
            x_natural, 
            epsilon=8/255, 
            BN_eval=True,
            random_init=True, 
            clip_min=0.0, 
            clip_max=1.0):

    criterion_kl = nn.KLDivLoss(reduction='none')
    if BN_eval : 
        model.eval()
    teacher_model.eval()

    if random_init:
        x_adv = x_natural + 0.001 * torch.randn(x_natural.shape).cuda()
    else:
        x_adv = x_natural
    x_adv = x_adv.detach().requires_grad_()  
    with torch.no_grad():
        teacher_output = F.softmax(teacher_model(x_adv), dim=1)
    with torch.enable_grad():
        loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), 
                                   teacher_output)
        loss_kl = torch.sum(loss_kl)
    grad = torch.autograd.grad(loss_kl, [x_adv])[0]
    x_adv = x_adv.detach() + epsilon * torch.sign(grad.detach())
    x_adv = torch.clamp(x_adv, clip_min, clip_max)

    x_adv = torch.clamp(x_adv, clip_min, clip_max).detach()
    return x_adv

def pgd_kl(model,
            teacher_model,
            x_natural,
            step_size=2/255,
            steps=10,
            epsilon=8/255,
            BN_eval=True,
            random_init=True,
            clip_min=0.0,
            clip_max=1.0):
    
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')
    if BN_eval:
        model.eval()

    # set eval mode for teacher model
    teacher_model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural + 0.001 * torch.randn(x_natural.shape).cuda()
    else:
        x_adv = x_natural
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_()
        with torch.no_grad():
            teacher_output = F.softmax(teacher_model(x_adv), dim=1)
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), 
                                   teacher_output)
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv], create_graph=False)[0].detach()
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    x_adv = torch.clamp(x_adv, clip_min, clip_max)
    return x_adv

