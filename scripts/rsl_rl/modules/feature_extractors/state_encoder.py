import copy

import torch
import torch.nn as nn


class StateHistoryEncoder(nn.Module):
    """对历史本体序列做 1D 卷积压缩，用于自监督对齐特权隐式。"""

    def __init__(self, activation_fn, input_size, tsteps, output_size, channel_size):
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3 * channel_size),
            self.activation_fn,
        )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(
                    in_channels=3 * channel_size,
                    out_channels=2 * channel_size,
                    kernel_size=8,
                    stride=4,
                ),
                self.activation_fn,
                nn.Conv1d(
                    in_channels=2 * channel_size, out_channels=channel_size, kernel_size=5, stride=1
                ),
                self.activation_fn,
                nn.Conv1d(
                    in_channels=channel_size, out_channels=channel_size, kernel_size=5, stride=1
                ),
                self.activation_fn,
                nn.Flatten(),
            )
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(
                    in_channels=3 * channel_size,
                    out_channels=2 * channel_size,
                    kernel_size=4,
                    stride=2,
                ),
                self.activation_fn,
                nn.Conv1d(
                    in_channels=2 * channel_size, out_channels=channel_size, kernel_size=2, stride=1
                ),
                self.activation_fn,
                nn.Flatten(),
            )
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(
                    in_channels=3 * channel_size,
                    out_channels=2 * channel_size,
                    kernel_size=6,
                    stride=2,
                ),
                self.activation_fn,
                nn.Conv1d(
                    in_channels=2 * channel_size, out_channels=channel_size, kernel_size=4, stride=2
                ),
                self.activation_fn,
                nn.Flatten(),
            )
        else:
            raise (ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
            nn.Linear(channel_size * 3, output_size), self.activation_fn
        )

    def forward(self, obs):
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1]))  # do projection for num_prop -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output


class _TemporalResidualBlock(nn.Module):
    """Small residual TCN block with causal-same padding for fixed-length history."""

    def __init__(self, channels: int, dilation: int, activation_fn: nn.Module):
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            ),
            copy.deepcopy(activation_fn),
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            ),
        )
        self.activation = copy.deepcopy(activation_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + x)


class TCNHistoryEncoder(nn.Module):
    """TCN encoder for student proprio history with configurable sequence length."""

    def __init__(self, activation_fn, input_size, tsteps, output_size, channel_size):
        super().__init__()
        self.tsteps = int(tsteps)
        if self.tsteps <= 0:
            raise ValueError(f"TCNHistoryEncoder expects a positive history length, got {tsteps}.")

        hidden_channels = max(int(channel_size), int(output_size), 32)
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_channels),
            copy.deepcopy(activation_fn),
        )
        self.temporal_backbone = nn.Sequential(
            _TemporalResidualBlock(hidden_channels, dilation=1, activation_fn=activation_fn),
            _TemporalResidualBlock(hidden_channels, dilation=2, activation_fn=activation_fn),
            _TemporalResidualBlock(hidden_channels, dilation=4, activation_fn=activation_fn),
        )
        self.output_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, output_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        projected = self.input_projection(obs.reshape(batch_size * self.tsteps, -1))
        temporal = projected.reshape(batch_size, self.tsteps, -1).permute(0, 2, 1)
        encoded = self.temporal_backbone(temporal)
        return self.output_head(encoded)


class TCNVelocityEstimator(nn.Module):
    """Dedicated TCN for estimating base linear velocity from proprio history."""

    def __init__(self, activation_fn, input_size, tsteps, output_size, channel_size):
        super().__init__()
        self.tsteps = int(tsteps)
        if self.tsteps <= 0:
            raise ValueError(f"TCNVelocityEstimator expects a positive history length, got {tsteps}.")

        hidden_channels = max(int(channel_size), 32)
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_channels),
            copy.deepcopy(activation_fn),
        )
        self.temporal_backbone = nn.Sequential(
            _TemporalResidualBlock(hidden_channels, dilation=1, activation_fn=activation_fn),
            _TemporalResidualBlock(hidden_channels, dilation=2, activation_fn=activation_fn),
            _TemporalResidualBlock(hidden_channels, dilation=4, activation_fn=activation_fn),
        )
        self.output_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            copy.deepcopy(activation_fn),
            nn.Linear(hidden_channels, output_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        projected = self.input_projection(obs.reshape(batch_size * self.tsteps, -1))
        temporal = projected.reshape(batch_size, self.tsteps, -1).permute(0, 2, 1)
        encoded = self.temporal_backbone(temporal)
        return self.output_head(encoded)
