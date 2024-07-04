import numpy as np
import torch
from tqdm import tqdm


class FDTD:
    C = 340.3
    AIR_DENSITY = 1.225
    BASE_PRESSURE = 0

    def __init__(
        self,
        box_dimensions,
        ds,
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
        self.grid_dimensions = (
            torch.round(torch.Tensor(box_dimensions) / self.ds).int().tolist()
        )
        self.pressure = torch.empty(self.grid_dimensions).to(dtype).to(self.device)
        self.vel_x = torch.empty(self.grid_dimensions).to(dtype).to(self.device)
        self.vel_y = torch.empty(self.grid_dimensions).to(dtype).to(self.device)
        self.vel_z = torch.empty(self.grid_dimensions).to(dtype).to(self.device)
        self.attenuation = torch.ones(self.grid_dimensions).to(dtype).to(self.device)
        self._make_pml(pml_layers)
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
        self.t = 0

    def iterate(self, iters=1, seconds=None, warm_start=True, show_progress=True):
        n_iters = round(seconds / self.dt) if seconds is not None else iters
        if not warm_start:
            self.reset()

        attenuation_dt = self.attenuation * self.dt + 1
        iter_range = tqdm(range(n_iters)) if show_progress else range(n_iters)
        with torch.no_grad():
            for _ in iter_range:
                self.vel_x -= self.vel_coef * (
                    torch.roll(self.pressure, 1, dims=0) - self.pressure
                )
                self.vel_y -= self.vel_coef * (
                    torch.roll(self.pressure, 1, dims=1) - self.pressure
                )
                self.vel_z -= self.vel_coef * (
                    torch.roll(self.pressure, 1, dims=2) - self.pressure
                )

                self.pressure -= self.pressure_coef * (
                    self.vel_x
                    - torch.roll(self.vel_x, -1, dims=0)
                    + self.vel_y
                    - torch.roll(self.vel_y, -1, dims=1)
                    + self.vel_z
                    - torch.roll(self.vel_z, -1, dims=2)
                )
                self.pressure /= attenuation_dt

                self.t += self.dt
