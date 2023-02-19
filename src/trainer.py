from tqdm import tqdm
import torch
import torch.nn as nn
from src.model import *
from torch.optim import AdamW
from src.logger import WanDBWriter


class Trainer:
    def __init__(self, dataloader, config, log):
        # self.config = config
        self.cfg = config
        self.device = self.cfg["training"]["device"]
        self.model = dict()
        self.K = 2 ** len(self.cfg["data"]["domains"])
        size = self.cfg["data"]["size"]
        self.model["gen"] = Generator(size=size)
        self.model["disc"] = Discriminator(K=self.K, size=size)
        self.model["map"] = MappingNetwork(K=self.K, D=self.cfg["model"]["D"])
        self.model["se"] = StyleEncoder(K=self.K, D=self.cfg["model"]["D"], size=size)

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

    def train(self):

        step = 0

        for epoch in range(self.cfg["training"]["epochs"]):
            for batch in tqdm(self.dataloader):
                # batch.to(self.device)
                real = batch[0].to(self.device)
                y_src = batch[1]["attributes"]

                step += 1

                self.logger.set_step(step) if self.log else None

                for block in self.model.values():
                    block.train()
                    block.to(self.device)

                self.optimizer_g.zero_grad()
                self.optimizer_d.zero_grad()
                self.optimizer_s.zero_grad()
                self.optimizer_m.zero_grad()

                z = torch.randn((self.cfg["training"]["batch_size"], 16)).to(
                    self.device
                )
                z2 = torch.randn((self.cfg["training"]["batch_size"], 16)).to(
                    self.device
                )
                # y_trg = torch.randint(
                #     size=(self.cfg["training"]["batch_size"], 1), low=0, high=self.K - 1
                # ).squeeze()
                y_trg = (
                    (
                        torch.randint(size=(1, 1), low=0, high=self.K)
                        * torch.ones(size=(self.cfg["training"]["batch_size"], 1))
                    )
                    .squeeze()
                    .long()
                )

                s = self.model["map"](z, y_trg)
                # s1_5 = self.model["map"](z, y_trg)  # shape of (B, num_domains, 16)
                s2 = self.model["map"](z2, y_trg)  # shape of (B, num_domains, 16)

                fake = self.model["gen"](real, s)
                # fake1_5 = self.model["gen"](real, s1_5)
                fake2 = self.model["gen"](real, s2)
                s_fake = self.model["se"](fake, y_trg)

                fake_reversed = self.model["gen"](fake, self.model["se"](real, y_src))

                d_real = self.model["disc"](real, y_src)
                d_fake_d = self.model["disc"](fake.detach(), y_trg)

                adv_fake_d = adversarial_loss(d_fake_d, 0)
                adv_real_d = adversarial_loss(d_real, 1)
                loss_d = adv_real_d + adv_fake_d

                loss_d.backward()
                self.optimizer_d.step()

                d_fake_g = self.model["disc"](fake, y_trg)
                adv_fake_g = adversarial_loss(d_fake_g, 1)

                style_rec_l = style_rec_loss(s, s_fake)
                style_div_l = style_div_loss(fake, fake2)
                cycle_l = cycle_loss(fake, fake_reversed)

                loss_g = adv_fake_g + style_rec_l + cycle_l  # - 2 * style_div_l

                loss_g.backward()

                self.optimizer_g.step()
                self.optimizer_s.step()
                self.optimizer_m.step()

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
                    self.log_images(fake_reversed, real)

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
