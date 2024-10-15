import torch
import torch.nn as nn
import torch.nn.functional as F
import memory
from base_adapter import BaseAdapter
from bn_layers import RobustBN1d, RobustBN2d
from AT import *

class RoTTA(BaseAdapter):
    def __init__(self, student, teacher, optimizer, attack=None):
        super(RoTTA, self).__init__(student, optimizer)
        self.mem = memory.CSTU(capacity=64, num_class=10, lambda_t=1.0, lambda_u=1.0)
        self.teacher = teacher
        self.update_frequency = 64  # actually the same as the size of memory bank
        self.current_instance = 0
        self.attack = attack
        self.ADaAD_Alpha = 0.5


    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        # batch data
        with torch.no_grad():
            model.eval()
            self.teacher.eval()
            ema_out = self.teacher(batch_data)
            predict = torch.softmax(ema_out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)

        # add into memory
        for i, data in enumerate(batch_data):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
            self.current_instance += 1

            if self.current_instance % self.update_frequency == 0:
                self.update_model(model, optimizer)

        return self.model(batch_data)

    def update_model(self, student, optimizer):
        x_nat, _ = self.mem.get_memory()
        x_nat = torch.stack(x_nat)
        x_adv = self.adaad_inner_loss(student, self.teacher, x_nat, self.attack)

        student.train()
        optimizer.zero_grad()

        ori_outputs = student(x_nat)
        adv_outputs = student(x_adv)

        with torch.no_grad():
            self.teacher.eval()
            t_ori_outputs = self.teacher(x_nat)
            t_adv_outputs = self.teacher(x_adv)

        kl_loss1 = nn.KLDivLoss()(F.log_softmax(adv_outputs, dim=1),
                                    F.softmax(t_adv_outputs.detach(), dim=1))
        kl_loss2 = nn.KLDivLoss()(F.log_softmax(ori_outputs, dim=1),
                                    F.softmax(t_ori_outputs.detach(), dim=1))
        
        loss = self.ADaAD_Alpha*kl_loss1 + (1-self.ADaAD_Alpha)*kl_loss2
        loss.backward()
        optimizer.step()

    def adaad_inner_loss(self, 
                     model,
                     teacher_model,
                     x_natural,
                     attack=None,
                     step_size=2/255,
                     steps=10,
                     epsilon=8/255,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0):
        
        if attack == 'FGSM' :
            return fgsm_kl(model, teacher_model, x_natural, epsilon, BN_eval, random_init, clip_min, clip_max)
        
        elif attack == 'PGD' :
            return pgd_kl(model, teacher_model, x_natural, step_size, steps, epsilon, BN_eval, random_init, clip_min, clip_max)

        return x_natural
        

    def configure_model(self, model: nn.Module):

        model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer, 0.05)
            momentum_bn.requires_grad_(True)
            set_named_submodule(model, name, momentum_bn)
        return model

def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)

def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))
