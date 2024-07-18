import numpy as np
import torch
from tqdm import tqdm


class FDTD:
    C = 340.3
    AIR_DENSITY = 1.225
    BASE_PRESSURE = 0
    DAMPING_COEF = 0.999

    def __init__(
        self,
        box_dimensions,
        ds,
        solid,
        device=None,
        cfl_factor=0.5,
        pml_layers=8,
        dtype=torch.float32,
    ):
        self.ds = ds
        self.device = (
            device
            if device is not None
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self.dt = cfl_factor * self.ds / (np.sqrt(2) * self.C)
        self.pressure_coef = self.AIR_DENSITY * self.C**2 * self.dt / self.ds
        self.vel_coef = self.dt / (self.ds * self.AIR_DENSITY)
        self.dtype = dtype
        self.box_dimensions = box_dimensions
        self.grid_dimensions = (
            torch.round(torch.as_tensor(box_dimensions) / self.ds).int().tolist()
        )
        self.pressure = torch.empty(
            self.grid_dimensions, dtype=dtype, device=self.device
        )
        self.vel_x = torch.empty(self.grid_dimensions, dtype=dtype, device=self.device)
        self.vel_y = torch.empty(self.grid_dimensions, dtype=dtype, device=self.device)
        self.vel_z = torch.empty(self.grid_dimensions, dtype=dtype, device=self.device)
        self.attenuation = torch.ones(
            self.grid_dimensions, dtype=dtype, device=self.device
        )
        self._make_pml(pml_layers)
        self.solid = solid
        self.emitter_amps = torch.zeros(
            self.grid_dimensions, dtype=dtype, device=self.device
        )
        self.emitter_freqs = torch.zeros(
            self.grid_dimensions, dtype=dtype, device=self.device
        )
        self.emitter_phases = torch.zeros(
            self.grid_dimensions, dtype=dtype, device=self.device
        )
        self.reset()

    def _make_pml(self, pml_layers):
        sigma_max = 0.5 / self.dt
        for i in range(pml_layers):
            j = None if i == 0 else -i
            self.attenuation[i, i:j, i:j] = self.attenuation[
                -1 - i, i:j, i:j
            ] = self.attenuation[i:j, i, i:j] = self.attenuation[
                i:j, -1 - i, i:j
            ] = self.attenuation[
                i:j, i:j, i
            ] = self.attenuation[
                i:j, i:j, -1 - i
            ] = sigma_max * (
                1 - i / pml_layers
            )

    def reset(self):
        self.pressure[...] = self.BASE_PRESSURE
        self.vel_x[...] = self.vel_y[...] = self.vel_z[...] = 0
        self.solid_damping = (
            (1 - torch.as_tensor(self.solid).to(self.device))
            * self.DAMPING_COEF
            / (self.attenuation * self.dt + 1)
        )
        self.t = 0

    def iterate(self, iters=1, seconds=None, warm_start=True, show_progress=True):
        n_iters = round(seconds / self.dt) if seconds is not None else iters
        if not warm_start:
            self.reset()

        attenuation_dt = self.attenuation * self.dt + 1
        iter_range = tqdm(range(n_iters)) if show_progress else range(n_iters)
        for _ in iter_range:
            self.vel_x -= self.vel_coef * (
                torch.roll(self.pressure, 1, dims=0) - self.pressure
            )
            self.vel_x *= self.solid_damping
            self.vel_y -= self.vel_coef * (
                torch.roll(self.pressure, 1, dims=1) - self.pressure
            )
            self.vel_y *= self.solid_damping
            self.vel_z -= self.vel_coef * (
                torch.roll(self.pressure, 1, dims=2) - self.pressure
            )
            self.vel_z *= self.solid_damping

            self.pressure -= self.pressure_coef * (
                self.vel_x
                - torch.roll(self.vel_x, -1, dims=0)
                + self.vel_y
                - torch.roll(self.vel_y, -1, dims=1)
                + self.vel_z
                - torch.roll(self.vel_z, -1, dims=2)
            )

            self.pressure /= attenuation_dt

            phase = torch.sin(
                2 * np.pi * self.emitter_freqs * self.t - self.emitter_phases
            ).to(self.device)
            self.pressure[self.emitter_amps != 0] = self.BASE_PRESSURE
            self.pressure += self.emitter_amps * phase

            self.t += self.dt

    def add_point_emitter(self, position, amp, freq, phase):
        assert (
            self.t == 0
        ), "Cannot add an emitter to a started simulation. Re-instantiate the object or call reset()"
        position = self._world_to_grid_coords(position)
        self.emitter_amps[*position] = amp
        self.emitter_freqs[*position] = freq
        self.emitter_phases[*position] = phase
        self.reset()
        return self

    def _world_to_grid_coords(self, position):
        return (
            (
                np.array(self.grid_dimensions)
                * np.array(position)
                / np.array(self.box_dimensions)
            )
            .round()
            .astype(int)
        )
