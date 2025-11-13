import torch
import torch.nn as nn
from contextlib import contextmanager
from models.networks import EDMPrecond
from models.ema import LitEma
from models.img_transformations import STFTEmbedder, DelayEmbedder

class ImagenTime(nn.Module):

    def __init__(self, args, device):
        super().__init__()
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.T = args.diffusion_steps
        self.device = device
        self.net = EDMPrecond(args.img_resolution, args.input_channels, channel_mult=args.ch_mult, model_channels=args.unet_channels, attn_resolutions=args.attn_resolution)
        if not args.use_stft:
            self.delay = args.delay
            self.embedding = args.embedding
            self.seq_len = args.seq_len
            self.ts_img = DelayEmbedder(self.device, args.seq_len, args.delay, args.embedding)
        else:
            self.ts_img = STFTEmbedder(self.device, args.seq_len, args.n_fft, args.hop_length)
        if args.ema:
            self.use_ema = True
            self.model_ema = LitEma(self.net, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup)
        else:
            self.use_ema = False

    def ts_to_img(self, signal, pad_val=None):
        return self.ts_img.ts_to_img(signal, True, pad_val) if pad_val else self.ts_img.ts_to_img(signal)

    def img_to_ts(self, img):
        return self.ts_img.img_to_ts(img)

    def init_stft_embedder(self, train_loader):
        assert type(self.ts_img) == STFTEmbedder, 'You must use the STFTEmbedder to initialize the min and max values'
        data = []
        for i, data_batch in enumerate(train_loader):
            data.append(data_batch[0])
        self.ts_img.cache_min_max_params(torch.cat(data, dim=0))

    def loss_fn(self, x):
        to_log = {}
        output, weight = self.forward(x)
        loss = (weight * (output - x).square()).mean()
        to_log['karras loss'] = loss.detach().item()
        return (loss, to_log)

    def loss_fn_impute(self, x, mask):
        to_log = {}
        output, weight = self.forward_impute(x, mask)
        x = self.unpad(x * (1 - mask), x.shape)
        output = self.unpad(output * (1 - mask), x.shape)
        loss = (weight * (output - x).square()).mean()
        to_log['karras loss'] = loss.detach().item()
        return (loss, to_log)

    def forward(self, x, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)
        n = torch.randn_like(y) * sigma
        D_yn = self.net(y + n, sigma, labels, augment_labels=augment_labels)
        return (D_yn, weight)

    def forward_impute(self, x, mask, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x) * sigma
        noise_impute = n * (1 - mask)
        x_to_impute = x * (1 - mask) + noise_impute
        x = x * mask
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)
        D_yn = self.net(y + x_to_impute, sigma, labels, augment_labels=augment_labels)
        return (D_yn, weight)

    def forward_forecast(self, past, future, labels=None, augment_pipe=None):
        s, e = (past.shape[-1], future.shape[-1])
        rnd_normal = torch.randn([past.shape[0], 1, 1, 1], device=past.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(past) if augment_pipe is not None else (past, None)
        n = torch.randn_like(future) * sigma
        full_seq = self.pad_f(torch.cat([past, future + n], dim=-1))
        D_yn = self.net(full_seq, sigma, labels, augment_labels=augment_labels)[..., s:s + e]
        return (D_yn, weight)

    def pad_f(self, x):
        _, _, cols, rows = x.shape
        max_side = max(32, rows)
        padding = (0, max_side - rows, 0, 0)
        x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=0)
        return x_padded

    def unpad(self, x, original_shape):
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f'{context}: Switched to EMA weights')
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())
                if context is not None:
                    print(f'{context}: Restored training weights')

    def on_train_batch_end(self, *args):
        if self.use_ema:
            self.model_ema(self.net)