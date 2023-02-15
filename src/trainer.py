class Trainer:
    def __init__(
        self,
        generator,
        discriminator,
        mapping_network,
        style_encoder,
        dataloader,
        optimizer_g,
        optimizer_d,
        optimizer_s,
        optimizer_m,
        scheduler_g,
        scheduler_d,
        config,
        device,
        logger,
    ):
        self.config = config
        self.device = device
        self.logger = logger

    def train(self):

        step = 0

        for epoch in range(self.config["training"]["epochs"]):
            for batch in tqdm(self.dataloader):
                step += 1
                self.logger.set_step(step)

                self.generator.train()
                self.discriminator.trian()
                self.mapping_network.train()
                self.style_encoder.train()

                z = torch.randn(16)
                z2 = torch.randn(16)
                s = self.mapping_network(z)  # shape of (B, num_domains, 16)
                s2 = self.mapping_network(z)  # shape of (B, num_domains, 16)
                real = batch["image"]
                fake = self.generator(real, s)
                fake2 = self.generator(real, s2)

                s_fake = self.style_encoder(fake)

                fake_reversed = self.generator(fake, self.style_encoder(real))

                d_real = self.discriminator(real)
                d_fake = self.discriminator(fake)

                adv_l = adversarial_loss(d_real, d_fake)
                style_rec_l = style_rec_loss(s, s_fake)
                style_div_l = style_div_loss(fake, fake2)
                cycle_l = cycle_loss(fake, fake_reversed)

                loss_g = -adv_l - style_rec_l + style_div_l - cycle_l
                loss_d = adv_l

                if step % self.config["training"]["log_steps"] == 0:
                    grad_norm_g = ...
                    grad_norm_d = ...
                    grad_norm_m = ...
                    grad_norm_s = ...

                    self.log_everything()

                    self.inference()

                if step % self.config["training"]["save_steps"] == 0:
                    save_checkpoint(...)

    def log_everything(
        self,
        step,
        epoch,
    ):
        self.logger.add_scalar("step", step)
        self.logger.add_scalar("epoch", epoch)

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
