from tqdm import tqdm
import torch
import torch.nn as nn
from src.model import *
from torch.optim import AdamW
from src.logger import WanDBWriter
import copy


class Trainer:
    def __init__(self, dataloader, config, log):
        # self.config = config
        self.cfg = config
        self.device = self.cfg["training"]["device"]
        self.model = dict()
        self.ema_model = dict()
        self.K = 2 ** len(self.cfg["data"]["domains"])
        size = self.cfg["data"]["size"]
        self.model["gen"] = Generator(size=size, D=self.cfg["model"]["D"])
        self.ema_model["gen"] = Generator(size=size, D=self.cfg["model"]["D"])
        self.model["disc"] = Discriminator(K=self.K, size=size)
        self.model["map"] = MappingNetwork(K=self.K, D=self.cfg["model"]["D"])
        self.ema_model["map"] = MappingNetwork(K=self.K, D=self.cfg["model"]["D"])
        self.model["se"] = StyleEncoder(K=self.K, D=self.cfg["model"]["D"], size=size)
        self.ema_model["se"] = StyleEncoder(
            K=self.K, D=self.cfg["model"]["D"], size=size
        )

        self.ema_model["se"] = copy.deepcopy(self.model["se"])
        self.ema_model["map"] = copy.deepcopy(self.model["map"])
        self.ema_model["gen"] = copy.deepcopy(self.model["gen"])

        # self.BS = BS
        # self.K = K
        # self.D = D
        # self.EPOCHS = EPOCHS
        self.log = log

        self.dataloader = dataloader
        self.optimizer_g = AdamW(
            self.model["gen"].parameters(), lr=self.cfg["training"]["lr"]
        )
        self.optimizer_d = AdamW(
            self.model["disc"].parameters(), lr=self.cfg["training"]["lr"]
        )
        self.optimizer_s = AdamW(
            self.model["se"].parameters(), lr=self.cfg["training"]["lr"]
        )
        self.optimizer_m = AdamW(
            self.model["map"].parameters(), lr=self.cfg["training"]["lr"]
        )

        self.logger = WanDBWriter(self.cfg) if self.log else None

        scheduler_g = None
        scheduler_d = None

    def discriminator_step(self, real, y_src, y_trg):
        real = real.requires_grad_()
        d_real = self.model["disc"](real, y_src)

        with torch.no_grad():
            z = torch.randn((self.cfg["training"]["batch_size"], 16)).to(self.device)
            s = self.model["map"](z, y_trg)
            fake = self.model["gen"](real, s)

        d_fake = self.model["disc"](fake, y_trg)

        return adversarial_loss(d_real, 1), adversarial_loss(d_fake, 0)

    def generator_step(self, real, y_src, y_trg):

        # ------------------------------------
        # --------- Adversarial Loss
        z = torch.randn((self.cfg["training"]["batch_size"], 16)).to(self.device)
        s = self.model["map"](z, y_trg)
        fake = self.model["gen"](real, s)
        adv_loss_g = adversarial_loss(self.model["disc"](fake, y_trg), 1)

        # ------------------------------------
        # --------- Cycle Loss
        fake_reversed = self.model["gen"](fake, self.model["se"](fake, y_src))
        c_loss = cycle_loss(fake_reversed, real)

        # ------------------------------------
        # --------- Style reconstruction Loss
        s_fake = self.model["se"](fake, y_trg)
        s_rec_loss = style_rec_loss(s, s_fake)
        # ------------------------------------
        # --------- Style divergence Loss
        z2 = torch.randn((self.cfg["training"]["batch_size"], 16)).to(self.device)
        s2 = self.model["map"](z2, y_trg)
        fake2 = self.model["gen"](real, s2)

        style_div_l = style_div_loss(fake, fake2)

        return adv_loss_g, c_loss, s_rec_loss, style_div_l, fake

    def train(self):

        step = 0

        for epoch in range(self.cfg["training"]["epochs"]):
            for batch in tqdm(self.dataloader):
                # batch.to(self.device)
                real = batch[0].to(self.device)
                y_src = batch[1]["attributes"].to(self.device)

                step += 1

                self.logger.set_step(step) if self.log else None

                for block in self.model.values():
                    block.train()
                    block.to(self.device)

                self.optimizer_g.zero_grad()
                self.optimizer_d.zero_grad()
                self.optimizer_s.zero_grad()
                self.optimizer_m.zero_grad()

                # y_trg = torch.randint(
                #     size=(self.cfg["training"]["batch_size"], 1), low=0, high=self.K - 1
                # ).squeeze()
                # y_trg = (
                #     (
                #         torch.randint(size=(1, 1), low=0, high=self.K)
                #         * torch.ones(size=(self.cfg["training"]["batch_size"], 1))
                #     )
                #     .squeeze()
                #     .long()
                # )
                y_trg = (
                    torch.ones(size=(self.cfg["training"]["batch_size"], 1)).squeeze()
                    - y_src
                ).long()

                assert all(y_trg != y_src)

                # -----------------------
                # ------ DISCRIMINATOR --
                adv_real_d, adv_fake_d = self.discriminator_step(real, y_src, y_trg)
                loss_d = adv_real_d + adv_fake_d
                loss_d.backward()
                self.optimizer_d.step()

                # -----------------------
                # ------ GENERATOR ------

                (
                    adv_fake_g,
                    cycle_l,
                    style_rec_l,
                    style_div_l,
                    fake,
                ) = self.generator_step(real, y_src, y_trg)
                loss_g = adv_fake_g + cycle_l + style_rec_l  # - 2 * style_div_l
                loss_g.backward()
                self.optimizer_g.step()
                self.optimizer_s.step()
                self.optimizer_m.step()

                self.moving_average(self.model["se"], self.ema_model["se"], 0.99)
                self.moving_average(self.model["se"], self.ema_model["se"], 0.99)
                self.moving_average(self.model["gen"], self.ema_model["gen"], 0.99)
                # self.soft_copy_param(self.ema_model["se"], self.model["se"], 0.99)
                # self.soft_copy_param(self.ema_model["map"], self.model["map"], 0.99)
                # self.soft_copy_param(self.ema_model["gen"], self.model["gen"], 0.99)

                if (self.log) and (step % self.cfg["training"]["log_steps"] == 0):
                    gnorm_g = self.get_grad_norm(self.model["gen"])
                    gnorm_d = self.get_grad_norm(self.model["disc"])
                    gnorm_m = self.get_grad_norm(self.model["map"])
                    gnorm_s = self.get_grad_norm(self.model["se"])
                    gnorms = [gnorm_g, gnorm_d, gnorm_m, gnorm_s]

                    self.log_scalars(
                        step,
                        epoch,
                        adv_fake_d.item(),
                        adv_fake_g.item(),
                        adv_real_d.item(),
                        loss_g.item(),
                        loss_d.item(),
                        cycle_l.item(),
                        style_div_l.item(),
                        style_rec_l.item(),
                        gnorms,
                    )
                    self.log_images(fake, real)

    def log_scalars(
        self,
        step,
        epoch,
        adv_fake_d,
        adv_fake_g,
        adv_real_d,
        loss_g,
        loss_d,
        cycle_l,
        style_div_l,
        style_rec_l,
        gnorms,
    ):
        self.logger.add_scalar("step", step)
        self.logger.add_scalar("epoch", epoch)
        self.logger.add_scalar("adv_fake_d", adv_fake_d)
        self.logger.add_scalar("adv_fake_g", adv_fake_g)
        self.logger.add_scalar("adv_real_d", adv_real_d)
        self.logger.add_scalar("loss_g", loss_g)
        self.logger.add_scalar("loss_d", loss_d)
        self.logger.add_scalar("loss_cycle", cycle_l)
        self.logger.add_scalar("loss_style_div", style_div_l)
        self.logger.add_scalar("loss_style_rex", style_rec_l)
        for grad_norm, label in zip(gnorms, ["G_gn", "D_gn", "M_gn", "SE_gn"]):
            self.logger.add_scalar(label, grad_norm)

    def log_images(self, fake, real):
        self.logger.add_image("fake", fake)
        self.logger.add_image("real", real)

    @torch.no_grad()
    def get_grad_norm(self, model, norm_type=2):
        """
        Move to utils
        """
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def soft_copy_param(target_link, source_link, tau):
        target_params = dict(target_link.namedparams())
        for param_name, param in source_link.namedparams():
            target_params[param_name].data[:] *= 1 - tau
            target_params[param_name].data[:] += tau * param.data

        # Soft-copy Batch Normalization's statistics
        target_links = dict(target_link.namedlinks())
        for link_name, link in source_link.namedlinks():
            if isinstance(link, L.BatchNormalization):
                target_bn = target_links[link_name]
                target_bn.avg_mean[:] *= 1 - tau
                target_bn.avg_mean[:] += tau * link.avg_mean
                target_bn.avg_var[:] *= 1 - tau
                target_bn.avg_var[:] += tau * link.avg_var

    def copy_param(self, target_link, source_link):
        target_params = dict(target_link.namedparams())
        for param_name, param in source_link.namedparams():
            target_params[param_name].data[:] = param.data

        # Copy Batch Normalization's statistics
        target_links = dict(target_link.namedlinks())
        for link_name, link in source_link.namedlinks():
            if isinstance(link, L.BatchNormalization):
                target_bn = target_links[link_name]
                target_bn.avg_mean[:] = link.avg_mean
                target_bn.avg_var[:] = link.avg_var

    def moving_average(self, model, model_test, beta=0.999):
        for param, param_test in zip(model.parameters(), model_test.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)
