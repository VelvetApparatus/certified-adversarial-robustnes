from collections import OrderedDict

import torch
import torch.nn.functional as F


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
        for key, old_w in model_state_dict.items():
            if key not in proxy_state_dict:
                continue
            if len(old_w.size()) <= 1:
                continue
            if "weight" not in key:
                continue

            new_w = proxy_state_dict[key]
            diff_w = new_w - old_w
            diff_norm = diff_w.norm()

            if diff_norm < self.weps:
                continue

            diff_dict[key] = old_w.norm() / (diff_norm + self.weps) * diff_w

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

        logits_clean = self.proxy(inputs_clean)
        logits_adv = self.proxy(inputs_adv)

        loss_natural = F.cross_entropy(logits_clean, targets)
        loss_robust = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_clean.detach(), dim=1),
            reduction="batchmean",
        )
        loss = - 1.0 * (loss_natural + beta * loss_robust)

        self.proxy_optim.zero_grad(set_to_none=True)
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = self.diff_in_weights()
        return diff

    def calc_awp_with_loss(self, loss_closure):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        self.proxy_optim.zero_grad(set_to_none=True)

        loss = -1.0 * loss_closure(self.proxy)
        loss.backward()
        self.proxy_optim.step()

        return self.diff_in_weights()

    def perturb(self, diff):
        self._add_into_weights(diff, coef=1.0)

    def restore(self, diff):
        self._add_into_weights(diff, coef=-1.0)


class AWPCrossEntropy(TradesAWP):
    def __init__(
            self,
            model,
            proxy,
            proxy_optim,
            wcoef,
            weps,
    ):
        super().__init__(
            model,
            proxy,
            proxy_optim,
            wcoef,
            weps,
        )


    def calc_awp(
            self,
            inputs_adv,
            targets,
            beta=None,
            inputs_clean=None,
    ):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        loss = -F.cross_entropy(self.proxy(inputs_adv), targets)
        self.proxy_optim.zero_grad(set_to_none=True)
        loss.backward()
        self.proxy_optim.step()

        return self.diff_in_weights()
