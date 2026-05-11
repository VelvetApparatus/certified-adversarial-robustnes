from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


class TradesAWP(object):
    def __init__(
            self,
            model,
            proxy,
            proxy_optim,
            wcoef,
            weps,

    ):
        super(TradesAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.wcoef = wcoef
        self.weps = weps

    def diff_in_weights(self):
        diff_dict = OrderedDict()
        model_state_dict = self.model.state_dict()
        proxy_state_dict = self.proxy.state_dict()
        for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
            if len(old_w.size()) <= 1:
                continue
            if 'weight' in old_k:
                diff_w = new_w - old_w
                diff_dict[old_k] = old_w.norm() / (diff_w.norm() + self.weps) * diff_w
        return diff_dict

    def _add_into_weights(
            self,
            diff,
            coef=1.0,
    ):
        wcoef = self.wcoef * coef
        names_in_diff = diff.keys()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in names_in_diff:
                    param.add_(wcoef * diff[name])

    def calc_awp(self, inputs_adv, inputs_clean, targets, beta):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        loss_natural = F.cross_entropy(self.proxy(inputs_clean), targets)
        loss_robust = F.kl_div(F.log_softmax(self.proxy(inputs_adv), dim=1),
                               F.softmax(self.proxy(inputs_clean), dim=1),
                               reduction='batchmean')
        loss = - 1.0 * (loss_natural + beta * loss_robust)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = self.diff_in_weights()
        return diff

    def perturb(self, diff):
        self._add_into_weights(diff, coef=1.0)

    def restore(self, diff):
        self._add_into_weights(diff, coef=-1.0)
