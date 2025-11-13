from omegaconf import OmegaConf
import argparse

def parse_args_uncond():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
    parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
    parser.add_argument('--log_dir', default='./logs', help='path to save logs')
    parser.add_argument('--neptune', type=bool, default=False, help='use neptune logger')
    parser.add_argument('--tags', type=str, default=['karras', 'unconditional'], help='tags for neptune logger', nargs='+')
    parser.add_argument('--beta1', type=float, default=1e-05, help='value of beta 1')
    parser.add_argument('--betaT', type=float, default=0.01, help='value of beta T')
    parser.add_argument('--deterministic', action='store_true', default=False, help='deterministic sampling')
    parser.add_argument('--config', type=str, default='./configs/unconditional/TS2I/fred_md.yaml', help='config file')
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='training batch size')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    parser.add_argument('--dataset', choices=['kdd_cup', 'traffic_hourly', 'solar_weekly', 'temperature_rain', 'nn5_daily', 'fred_md', 'sine', 'energy', 'time_series', 'stocks'], help='training dataset')
    parser.add_argument('--seq_len', type=int, help='input sequence length, only needed if using short-term datasets(stocks,sine,energy,time_series)')
    parser.add_argument('--use_stft', type=bool, help='use stft transform - if absent, use delay embedding')
    parser.add_argument('--n_fft', type=int, help='n_fft, only needed if using stft')
    parser.add_argument('--hop_length', type=int, help='hop_length, only needed if using stft')
    parser.add_argument('--delay', type=int, help='delay for the delay embedding transformation, only needed if using delay embedding')
    parser.add_argument('--embedding', type=int, help='embedding for the delay embedding transformation, only needed if using delay embedding')
    parser.add_argument('--img_resolution', type=int, help='image resolution')
    parser.add_argument('--input_channels', type=int, help='number of image channels, 2 if stft is used, 1 for delay embedding')
    parser.add_argument('--unet_channels', type=int, help='number of unet channels')
    parser.add_argument('--ch_mult', type=int, help='ch mut', nargs='+')
    parser.add_argument('--attn_resolution', type=int, help='attn_resolution', nargs='+')
    parser.add_argument('--diffusion_steps', type=int, help='number of diffusion steps')
    parser.add_argument('--ema', type=bool, help='use ema')
    parser.add_argument('--ema_warmup', type=int, help='ema warmup')
    parser.add_argument('--logging_iter', type=int, default=100, help='number of iterations between logging')
    parser.add_argument('--percent', type=int, default=100)
    parsed_args = parser.parse_args()
    config = OmegaConf.to_object(OmegaConf.load(parsed_args.config))
    for k, v in vars(parsed_args).items():
        if v is None:
            setattr(parsed_args, k, config.get(k, None))
    for k, v in config.items():
        if k not in vars(parsed_args):
            setattr(parsed_args, k, v)
    if parsed_args.dataset in ['stock', 'sine', 'energy', 'time_series']:
        parsed_args.input_size = parsed_args.input_channels
    return parsed_args

def parse_args_cond():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
    parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
    parser.add_argument('--log_dir', default='./logs', help='path to save logs')
    parser.add_argument('--neptune', type=bool, default=False, help='use neptune logger')
    parser.add_argument('--tags', type=str, default=['karras', 'conditional'], help='tags for neptune logger', nargs='+')
    parser.add_argument('--beta1', type=float, default=1e-05, help='value of beta 1')
    parser.add_argument('--betaT', type=float, default=0.01, help='value of beta T')
    parser.add_argument('--deterministic', action='store_true', default=False, help='deterministic sampling')
    parser.add_argument('--config', type=str, default='./configs/interpolation/TS2I/physionet.yaml', help='config file')
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='training batch size')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    parser.add_argument('--dataset', choices=['kdd_cup', 'traffic_hourly', 'solar_weekly', 'temperature_rain', 'nn5_daily', 'fred_md', 'sine', 'energy', 'time_series', 'stocks'], help='training dataset')
    parser.add_argument('--seq_len', type=int, help='input sequence length, only needed if using short-term datasets(stocks,sine,energy,time_series)')
    parser.add_argument('--use_stft', type=bool, help='use stft transform - if absent, use delay embedding')
    parser.add_argument('--n_fft', type=int, help='n_fft, only needed if using stft')
    parser.add_argument('--hop_length', type=int, help='hop_length, only needed if using stft')
    parser.add_argument('--delay', type=int, help='delay for the delay embedding transformation, only needed if using delay embedding')
    parser.add_argument('--embedding', type=int, help='embedding for the delay embedding transformation, only needed if using delay embedding')
    parser.add_argument('--img_resolution', type=int, help='image resolution')
    parser.add_argument('--input_channels', type=int, help='number of image channels, 2 if stft is used, 1 for delay embedding')
    parser.add_argument('--unet_channels', type=int, help='number of unet channels')
    parser.add_argument('--ch_mult', type=int, help='ch mut', nargs='+')
    parser.add_argument('--attn_resolution', type=int, help='attn_resolution', nargs='+')
    parser.add_argument('--diffusion_steps', type=int, help='number of diffusion steps')
    parser.add_argument('--ema', type=bool, help='use ema')
    parser.add_argument('--ema_warmup', type=int, help='ema warmup')
    parser.add_argument('--logging_iter', type=int, default=100, help='number of iterations between logging')
    parser.add_argument('--percent', type=int, default=100)
    parsed_args = parser.parse_args()
    config = OmegaConf.to_object(OmegaConf.load(parsed_args.config))
    for k, v in vars(parsed_args).items():
        if v is None:
            setattr(parsed_args, k, config.get(k, None))
    for k, v in config.items():
        if k not in vars(parsed_args):
            setattr(parsed_args, k, v)
    return parsed_args