import torch
import torchaudio


class STFTDeltaFrontend(torch.nn.Module):
    def __init__(
        self,
        n_fft: int,
        sample_rate: int,
        win_length_ms: int,
        hop_length_ms: int,
        delta_win: int,
        delta_delta_win: int,
    ):
        super().__init__()
        win_length = int(win_length_ms * sample_rate / 1000)
        hop_length = int(hop_length_ms * sample_rate / 1000)
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length
        )
        self.deltas = torchaudio.transforms.ComputeDeltas(win_length=delta_win)
        self.deltasdeltas = torchaudio.transforms.ComputeDeltas(
            win_length=delta_delta_win
        )

    def forward(self, data_object: torch.Tensor, **batch):
        data_object = data_object.squeeze(1)
        # I did not have a good experience with log due to gradient nans
        spec = self.spectrogram(data_object)
        deltas = self.deltas(spec)
        deltasdeltas = self.deltasdeltas(deltas)
        input = torch.stack((spec, deltas, deltasdeltas), dim=1)
        return input
