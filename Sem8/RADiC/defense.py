import torch
import torch.nn as nn
import torch.nn.functional as F

from RADiC.modules.base_adapter import BaseAdapter, softmax_entropy
from RADiC.modules.memory import CSTU
from RADiC.modules.bn_layers import RobustBN1d, RobustBN2d

class RoTTA(BaseAdapter):
    def __init__(self, model, model_ema, optimizer, attack='FGSM', args=None):
        super(RoTTA, self).__init__(model, optimizer)

        if model_ema == None :
            self.model_ema = self.build_ema(self.model)
        else :
            self.model_ema = model_ema

        self.mem = CSTU(capacity=64, num_class=10, lambda_t=1.0, lambda_u=1.0)
        self.nu = 0.001
        self.update_frequency = 64  # actually the same as the size of memory bank
        self.current_instance = 0
        self.attack = attack
        self.args = args

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        # batch data
        with torch.no_grad():
            model.eval()
            self.model_ema.eval()
            ema_out = self.model_ema(batch_data)
            predict = torch.softmax(ema_out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = torch.sum( -predict * torch.log(predict + 1e-6), dim=1)

        # add into memory
        for i, data in enumerate(batch_data):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
            self.current_instance += 1

            if self.current_instance % self.update_frequency == 0:
                self.update_model(model, optimizer)

        if self.args.kd == True :
            return self.model(batch_data)
        return ema_out

    def update_model(self, model, optimizer):
        model.train()
        self.model_ema.train()
        l = self.calculate_loss()
        if l is not None:
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        self.update_ema_variables(self.model_ema, self.model, self.nu)

    def calculate_loss(self):
        sup_data, ages = self.mem.get_memory()

        if len(sup_data) == 0 :
            return None

        if self.args.kd == True :
            alpha = 0.5
            x_nat = torch.stack(sup_data)
            x_adv = self.adaad_inner_loss(self.model, self.model_ema, x_nat, self.attack)

            ori_outputs = self.model(x_nat)
            adv_outputs = self.model(x_adv)

            with torch.no_grad():
                self.teacher.eval()
                t_ori_outputs = self.model_ema(x_nat)
                t_adv_outputs = self.model_ema(x_adv)

            kl_loss1 = nn.KLDivLoss()(F.log_softmax(adv_outputs, dim=1),
                                        F.softmax(t_adv_outputs.detach(), dim=1))
            kl_loss2 = nn.KLDivLoss()(F.log_softmax(ori_outputs, dim=1),
                                        F.softmax(t_ori_outputs.detach(), dim=1))

            loss = alpha*kl_loss1 + (1-alpha)*kl_loss2

        else :

            sup_data = torch.stack(sup_data)
            ema_sup_out = self.model_ema(sup_data)
            stu_sup_out = self.model(sup_data)
            instance_weight = timeliness_reweighting(ages)
            entropy_loss = (softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()
            discrepancy_loss = 0

            if self.args.arch == 'radic' :
              fitness_lambda = 0.2
              features = self.feature_extractor(sup_data)
              features = torch.flatten(features, 1)
              batch_std, batch_mean = torch.std_mean(features, dim=0)
              criterion_mse = torch.nn.MSELoss(reduction='none').cuda()
              std_mse = criterion_mse(batch_std, self.train_info['std'])
              mean_mse = criterion_mse(batch_mean, self.train_info['mean'])
              discrepancy_loss = fitness_lambda * (std_mse.sum() + mean_mse.sum()) * sup_data.shape[0] / 64

            loss = discrepancy_loss + entropy_loss

        return loss


    def adaad_inner_loss(self, model, teacher_model, x_natural, attack=None,
                         step_size=2/255, steps=10, epsilon=8/255,
                         BN_eval=True, random_init=True, clip_min=0.0,
                         clip_max=1.0):

        if attack == 'FGSM' :
            return fgsm_kl(model, teacher_model, x_natural, epsilon, BN_eval, random_init, clip_min, clip_max)

        elif attack == 'PGD' :
            return pgd_kl(model, teacher_model, x_natural, step_size, steps, epsilon, BN_eval, random_init, clip_min, clip_max)

        return x_natural

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
            return ema_model

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

    def obtain_origin_stat(self, train_loader):
        print('===> begin calculating mean and variance for ResNet18')
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1]).cuda()
        features = []
        with torch.no_grad():
            for images, _ in train_loader:
                images = images.cuda()
                # Pass images through the model to get features
                feature = self.feature_extractor(images)
                feature = torch.flatten(feature, 1)
                features.append(feature)  # Flatten the features

            features = torch.cat(features, dim=0)
            std, mean = torch.std_mean(features, dim=0)
            self.train_info = {'std': std, 'mean': mean}
        del features
        print('===> calculating mean and variance for ResNet18 end')


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
