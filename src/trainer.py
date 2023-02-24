from tqdm import tqdm
import torch
import torch.nn as nn
from src.model import *
from torch.optim import AdamW
from src.logger import WanDBWriter
import copy
import os
from lpips_pytorch import LPIPS
from sklearn.metrics import roc_auc_score

# from torchmetrics.image.fid import FrechetInceptionDistance


class Trainer:
    def __init__(self, dataloader, val_dataloader, reference_dataloader, config, log):
        # self.config = config
        self.cfg = config
        self.device = self.cfg["training"]["device"]
        self.model = dict()
        self.avg_model = dict()
        self.K = 2 ** len(self.cfg["data"]["domains"])
        size = self.cfg["data"]["size"]
        self.model["gen"] = Generator(size=size, D=self.cfg["model"]["D"])
        self.avg_model["gen"] = Generator(size=size, D=self.cfg["model"]["D"])
        self.model["disc"] = Discriminator(K=self.K, size=size)
        self.model["map"] = MappingNetwork(K=self.K, D=self.cfg["model"]["D"])
        self.avg_model["map"] = MappingNetwork(K=self.K, D=self.cfg["model"]["D"])
        self.model["se"] = StyleEncoder(K=self.K, D=self.cfg["model"]["D"], size=size)
        self.avg_model["se"] = StyleEncoder(
            K=self.K, D=self.cfg["model"]["D"], size=size
        )

        self.r1 = R1(w=1)

        # for _, model in self.model.items():
        #     model.apply(self.init_weights)

        # for _, model in self.avg_model.items():
        #     model.apply(self.init_weights)

        self.avg_model["se"] = copy.deepcopy(self.model["se"]).to(self.device)
        self.avg_model["map"] = copy.deepcopy(self.model["map"]).to(self.device)
        self.avg_model["gen"] = copy.deepcopy(self.model["gen"]).to(self.device)

        self.log = log

        self.dataloader = dataloader
        self.ref_dataloader = reference_dataloader
        self.val_dataloader = val_dataloader
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
            self.model["map"].parameters(), lr=self.cfg["training"]["lr"] / 10
        )

        self.logger = WanDBWriter(self.cfg) if self.log else None
        # self.logger.add_text("arch_map", self.model["map"].__repr__())
        # self.logger.add_text("arch_d", self.model["disc"].__repr__())
        # self.logger.add_text("arch_g", self.model["gen"].__repr__())
        # self.logger.add_text("arch_se", self.model["se"].__repr__())

        # for _, model in self.model.items():
        #     model.apply(self.init_weights)

        scheduler_g = None
        scheduler_d = None

    def init_weights(self, weight):
        if isinstance(weight, nn.Conv2d):
            nn.init.kaiming_normal_(weight.weight, mode="fan_in", nonlinearity="relu")
            if weight.bias is not None:
                nn.init.constant_(weight.bias, 0)
        if isinstance(weight, nn.Linear):
            nn.init.kaiming_normal_(weight.weight, mode="fan_in", nonlinearity="relu")
            if weight.bias is not None:
                nn.init.constant_(weight.bias, 0)

    def eval(self):

        # criterion = LPIPS(net_type="alex", version="0.1")
        # criterion = criterion.to(self.device)
        # # fid = FrechetInceptionDistance(feature=64)
        # n_examples = 10

        # with torch.no_grad():
        #     lpipses = []
        #     for batch in self.val_dataloader:
        #         real = batch[0].to(self.device)
        #         y_src = batch[1]["attributes"].to(self.device)
        #         y_trg = (
        #             torch.ones(size=(self.cfg["training"]["batch_size"] * 2, 1))
        #             .squeeze()
        #             .to(self.device)
        #             - y_src
        #         ).long()

        #         zs_trg = torch.randn(
        #             (n_examples, self.cfg["training"]["batch_size"] * 2, 16)
        #         ).to(
        #             self.device
        #         )  # num_samples x batch x  size

        #         # нужно в стайл энкодер подавать что-то с первой размерностью в виде батчсайза

        #         styles = [self.avg_model["map"](z, y_trg) for z in zs_trg]
        #         fakes = torch.stack([self.avg_model["gen"](real, s) for s in styles])
        #         # self.avg_model["gen"](real, s) (B, 3,H,W)
        #         fakes = fakes.permute(1, 0, 2, 3, 4)
        #         # self.avg_model["gen"](real, s) (B, 3,H,W, 10)
        #         for reference_set in fakes:
        #             for i, im1 in enumerate(reference_set):
        #                 for j, im2 in enumerate(reference_set):
        #                     if i > j:
        #                         lpipses.append(criterion(im1, im2))
        #     metric = torch.mean(torch.stack(lpipses))

        self.logger.add_scalar("lpips_val_latent", 0)

    def discriminator_step(self, real, y_src, y_trg):
        real = real.requires_grad_()
        d_real = self.model["disc"](real, y_src)

        with torch.no_grad():
            z = torch.randn((self.cfg["training"]["batch_size"], 16)).to(self.device)
            s = self.model["map"](z, y_trg)
            fake = self.model["gen"](real, s)

        d_fake = self.model["disc"](fake, y_trg)
        # with torch.no_grad():
        #     d_fake2 = self.model["disc"](real, (1 - y_src).long())
        # print(torch.mean(d_real - d_fake2).item())

        r_loss = self.r1(d_real, real)

        real_loss = adversarial_loss(d_real, 1)
        fake_loss = adversarial_loss(d_fake, 0)

        return (
            real_loss + fake_loss,
            r_loss,
        )

    def checkpoint(self, epoch):
        path = self.cfg["data"]["checkpoint"]
        os.makedirs(f"{path}epoch_{epoch}", exist_ok=True)
        path = path + f"epoch_{epoch}"
        for name, model in self.model.items():
            torch.save(model.state_dict(), path + f"/{name}.pth")
        torch.save(self.optimizer_g.state_dict(), path + "/opt_g.pth")
        torch.save(self.optimizer_d.state_dict(), path + "/opt_d.pth")
        torch.save(self.optimizer_s.state_dict(), path + "/opt_s.pth")
        torch.save(self.optimizer_m.state_dict(), path + "/opt_m.pth")

    def rec_step(self, real, y_src, batch_ref):
        real = real.requires_grad_()

        real_ref = batch_ref[0].to(self.device)

        # assert torch.sum(torch.abs(real_ref[0] - real[0])) > 0
        # print(torch.sum(torch.abs(real_ref[0] - real[0])))

        real_ref2 = real_ref.flip((0,))
        y_ref = batch_ref[1]["attributes"].to(self.device)
        y_ref2 = y_ref.flip((0,))

        real_ref = real_ref.requires_grad_()

        s_ref = self.model["se"](real_ref, y_ref)
        fake = self.model["gen"](real, s_ref)

        adv_loss_g = adversarial_loss(self.model["disc"](fake, y_ref), 1)
        fake_reversed = self.model["gen"](fake, s_ref)
        cycle_loss_g = cycle_loss(real_ref, fake_reversed)
        style_loss_g = style_rec_loss(s_ref, self.model["se"](fake, y_ref))

        s_ref2 = s_ref = self.model["se"](real_ref2, y_ref2)
        s_div_loss = style_div_loss(fake, self.model["gen"](real, s_ref2).detach())

        fake = self.model["gen"](real, s_ref)
        d_fake = self.model["disc"](fake.detach(), y_ref)
        d_real = self.model["disc"](real_ref, y_ref)
        r_loss = self.r1(d_real, real_ref)
        adv_loss_d = adversarial_loss(d_fake, 0) + adversarial_loss(d_real, 1) + r_loss

        return (
            adv_loss_g,
            cycle_loss_g,
            style_loss_g,
            s_div_loss,
            adv_loss_d,
        )

    def generator_step(self, real, y_src, y_trg):
        real = real.requires_grad_()

        # ------------------------------------
        # --------- Adversarial Loss
        z = torch.randn((self.cfg["training"]["batch_size"], 16)).to(self.device)
        s = self.model["map"](z, y_trg)
        fake = self.model["gen"](real, s)
        adv_loss_g = F.binary_cross_entropy_with_logits(
            self.model["disc"](fake, y_trg), torch.ones_like(y_trg) * 0.9
        )

        # ------------------------------------
        # --------- Cycle Loss
        fake_reversed = self.model["gen"](fake, self.model["se"](real, y_src))
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
        style_div_l = style_div_loss(fake, fake2.detach())

        return (
            adv_loss_g,
            c_loss,
            s_rec_loss,
            style_div_l,
            fake,
            fake_reversed,
        )

    def train(self):

        step = 0
        for block in self.model.values():
            block.train()
            block.to(self.device)

        for epoch in range(self.cfg["training"]["epochs"]):
            for batch, batch_ref in tqdm(zip(self.dataloader, self.ref_dataloader)):
                # batch.to(self.device)
                real = batch[0].to(self.device)
                y_src = batch[1]["attributes"].to(self.device)

                step += 1

                self.logger.set_step(step) if self.log else None

                self.optimizer_g.zero_grad()
                self.optimizer_d.zero_grad()
                self.optimizer_s.zero_grad()
                self.optimizer_m.zero_grad()

                if self.cfg["training"]["random_target"]:
                    y_trg = (
                        torch.randint(
                            size=(self.cfg["training"]["batch_size"], 1), low=0, high=2
                        )
                        .squeeze()
                        .to(self.device)
                    ).long()

                else:
                    y_trg = (
                        torch.ones(size=(self.cfg["training"]["batch_size"], 1))
                        .squeeze()
                        .to(self.device)
                        - y_src
                    ).long()

                # -----------------------
                # ------ DISCRIMINATOR --
                loss_disc, r_loss = self.discriminator_step(real, y_src, y_trg)
                loss_d = loss_disc + r_loss
                loss_d.backward()
                self.optimizer_d.step()

                # --------------------------
                # ------ GENERATOR Latent --

                (
                    adv_fake_g,
                    cycle_l,
                    style_rec_l,
                    style_div_l,
                    fake,
                    fake_rec,
                ) = self.generator_step(real, y_src, y_trg)

                loss_g = (
                    adv_fake_g + style_rec_l + cycle_l  # - style_div_l * (0.9997**step)
                )
                loss_g.backward()
                self.optimizer_g.step()
                self.optimizer_s.step()
                self.optimizer_m.step()

                # -------------------------
                # ------ GENERATOR REF ----

                (
                    adv_loss_g_ref,
                    cycle_loss_g_ref,
                    style_loss_g_ref,
                    style_div_loss_ref,
                    adv_loss_d_ref,
                ) = self.rec_step(real, y_src, batch_ref)

                loss_g_ref = (
                    adv_loss_g_ref
                    + cycle_loss_g_ref
                    + style_loss_g_ref
                    - style_div_loss_ref * (0.9997**step)
                )
                loss_d_ref = adv_loss_d_ref

                self.optimizer_g.zero_grad()
                self.optimizer_d.zero_grad()

                loss_g_ref.backward()
                loss_d_ref.backward()

                self.optimizer_g.step()
                self.optimizer_d.step()

                # Exponential moving average for validation
                # self.ema_weight_averaging(self.model["se"], self.avg_model["se"], 0.999)
                # self.ema_weight_averaging(self.model["se"], self.avg_model["se"], 0.999)
                # self.ema_weight_averaging(
                #     self.model["gen"], self.avg_model["gen"], 0.999
                # )

                if (self.log) and (step % self.cfg["training"]["log_steps"] == 0):
                    self.eval()
                    gnorm_g = self.get_grad_norm(self.model["gen"])
                    gnorm_d = self.get_grad_norm(self.model["disc"])
                    gnorm_m = self.get_grad_norm(self.model["map"])
                    gnorm_s = self.get_grad_norm(self.model["se"])
                    gnorms = [gnorm_g, gnorm_d, gnorm_m, gnorm_s]

                    self.log_scalars(
                        step,
                        epoch,
                        # adv_fake_d.item(),
                        adv_fake_g.item(),
                        # adv_real_d.item(),
                        loss_g.item(),
                        loss_g_ref.item(),
                        loss_d.item(),
                        loss_d_ref.item(),
                        cycle_l.item(),
                        style_div_l.item(),
                        style_rec_l.item(),
                        gnorms,
                        adv_loss_g_ref.item(),
                        cycle_loss_g_ref.item(),
                        style_loss_g_ref.item(),
                        style_div_loss_ref.item(),
                    )
                    self.log_images(fake, fake_rec, real)
                    self.generate_samples_from_reference()
                    # self.test_dls()
            self.checkpoint(epoch)

    # def test_dls(self):
    #     refs = next(iter(self.val_dataloader))

    #     imgs = refs[0].to(self.device)
    #     y = refs[1]["attributes"].to(self.device)
    #     male_y = y[y == 1]
    #     male_img = imgs[y == 1]
    #     female_y = y[y == 0]
    #     female_img = imgs[y == 0]
    #     self.logger.add_image("images_label_0", female_img)
    #     self.logger.add_image("images_label_1", male_img)
    #     self.logger.add_image("all_images_label_1", imgs)
    #     self.logger.add_text("all_labels", f"{y}")

    def generate_samples_from_reference(self):
        with torch.no_grad():

            refs = next(iter(self.val_dataloader))
            imgs = refs[0].to(self.device)
            y = refs[1]["attributes"].to(self.device)
            # male_y = y[y == 1][0]
            male_img = imgs[y == 1][0].unsqueeze(0)
            # female_y = y[y == 0][0]
            female_img = imgs[y == 0][0].unsqueeze(0)

            s_ref = self.model["se"](male_img, [1])
            fake = self.model["gen"](female_img, s_ref)

            z1 = torch.randn((1, 16)).to(self.device)
            z2 = torch.randn((1, 16)).to(self.device)
            s1 = self.model["map"](z1, [0])
            s2 = self.model["map"](z2, [0])
            s3 = self.model["map"](z1, [1])

            fake_female = self.model["gen"](male_img, s1)
            fake_female2 = self.model["gen"](male_img, s2)
            fake_male = self.model["gen"](male_img, s3)

        self.logger.add_image("male_to_female", fake)
        self.logger.add_image("src_male", male_img)
        self.logger.add_image("ref_female", female_img)
        self.logger.add_image("male_to_female_z1", fake_female)
        self.logger.add_image("male_to_female_z2", fake_female2)
        self.logger.add_image("male_to_male_z1", fake_male)

    def log_scalars(
        self,
        step,
        epoch,
        # adv_fake_d,
        adv_fake_g,
        # adv_real_d,
        loss_g,
        loss_g_ref,
        loss_d,
        loss_d_ref,
        cycle_l,
        style_div_l,
        style_rec_l,
        gnorms,
        adv_loss_g_ref,
        cycle_loss_g_ref,
        style_loss_g_ref,
        style_div_loss_ref,
    ):
        self.logger.add_scalar("step", step)
        self.logger.add_scalar("epoch", epoch)
        # self.logger.add_scalar("adv_fake_d", adv_fake_d)
        self.logger.add_scalar("adv_fake_g", adv_fake_g)
        # self.logger.add_scalar("adv_real_d", adv_real_d)
        self.logger.add_scalar("loss_g", loss_g)
        self.logger.add_scalar("loss_g_ref", loss_g_ref)
        self.logger.add_scalar("loss_d", loss_d)
        self.logger.add_scalar("loss_d_ref", loss_d_ref)
        self.logger.add_scalar("loss_cycle", cycle_l)
        self.logger.add_scalar("loss_style_div", style_div_l)
        self.logger.add_scalar("loss_style_rex", style_rec_l)
        self.logger.add_scalar("adv_loss_g_ref", adv_loss_g_ref)
        self.logger.add_scalar("cycle_loss_g_ref", cycle_loss_g_ref)
        self.logger.add_scalar("style_loss_g_ref", style_loss_g_ref)
        self.logger.add_scalar("style_div_loss_ref", style_div_loss_ref)
        for grad_norm, label in zip(gnorms, ["G_gn", "D_gn", "M_gn", "SE_gn"]):
            self.logger.add_scalar(label, grad_norm)

    def log_images(self, fake, fake_rec, real):
        self.logger.add_image("fake", fake)
        self.logger.add_image("fake_rec", fake_rec)
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

    def ema_weight_averaging(self, model, model_ema, beta=0.999):
        for param, param_test in zip(model.parameters(), model_ema.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)
