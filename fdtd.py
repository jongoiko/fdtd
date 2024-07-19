import numpy as np
import torch
import skimage.draw
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
        self.amplitude = torch.zeros(
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
        self.emitter_angles = torch.zeros(
            self.grid_dimensions + [3], dtype=dtype, device=self.device
        )
        self.is_point_emitter = torch.zeros(
            self.grid_dimensions, dtype=bool, device=self.device
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
        self.amplitude[...] = 0
        self.t = 0

    def iterate(
        self,
        iters=1,
        seconds=None,
        warm_start=True,
        show_progress=True,
        amp_measurement_warmup=None,
    ):
        if seconds is not None:
            iters = round(seconds / self.dt)
            if amp_measurement_warmup is not None:
                amp_measurement_warmup = round(amp_measurement_warmup / self.dt)
        if not warm_start:
            self.reset()

        attenuation_dt = self.attenuation * self.dt + 1
        iter_range = tqdm(range(iters)) if show_progress else range(iters)
        for i in iter_range:
            self.vel_x -= self.vel_coef * (
                torch.roll(self.pressure, -1, dims=0) - self.pressure
            )
            self.vel_x *= self.solid_damping
            self.vel_y -= self.vel_coef * (
                torch.roll(self.pressure, -1, dims=1) - self.pressure
            )
            self.vel_y *= self.solid_damping
            self.vel_z -= self.vel_coef * (
                torch.roll(self.pressure, -1, dims=2) - self.pressure
            )
            self.vel_z *= self.solid_damping

            phase = torch.sin(
                2 * np.pi * self.emitter_freqs * self.t - self.emitter_phases
            ).to(self.device)

            directed_emitter_amps = self.emitter_amps.where(
                self.is_point_emitter.logical_not(), 0.0
            )
            self.vel_x *= 1 - directed_emitter_amps
            self.vel_x += directed_emitter_amps * self.emitter_angles[..., 0] * phase
            self.vel_y *= 1 - directed_emitter_amps
            self.vel_y += directed_emitter_amps * self.emitter_angles[..., 1] * phase
            self.vel_z *= 1 - directed_emitter_amps
            self.vel_z += directed_emitter_amps * self.emitter_angles[..., 2] * phase

            self.pressure -= self.pressure_coef * (
                self.vel_x
                - torch.roll(self.vel_x, 1, dims=0)
                + self.vel_y
                - torch.roll(self.vel_y, 1, dims=1)
                + self.vel_z
                - torch.roll(self.vel_z, 1, dims=2)
            )

            self.pressure /= attenuation_dt

            self.pressure[self.emitter_amps != 0] = self.BASE_PRESSURE
            self.pressure += (self.emitter_amps * phase).where(
                self.is_point_emitter, 0.0
            )

            if amp_measurement_warmup is not None and i >= amp_measurement_warmup:
                self._get_amp()

            self.t += self.dt

    def _get_amp(self):
        self.amplitude = torch.maximum(self.amplitude, torch.abs(self.pressure))

    def _assert_can_add_emitter(self):
        assert (
            self.t == 0
        ), "Cannot add an emitter to a started simulation. Re-instantiate the object or call reset()"

    def add_point_emitter(self, position, amp, freq, phase):
        self._assert_can_add_emitter()
        position = self.world_to_grid_coords(position)
        self.emitter_amps[*position] = amp
        self.emitter_freqs[*position] = freq
        self.emitter_phases[*position] = phase
        self.is_point_emitter[*position] = True
        return self

    def add_circular_emitter(self, position, amp, freq, phase, angle, radius):
        self._assert_can_add_emitter()
        normal = np.array(
            [
                np.sin(angle[0]),
                np.cos(angle[0]) * np.cos(np.pi / 2 + angle[1]),
                np.cos(angle[0]) * np.cos(angle[1]),
            ]
        )
        basis_1, basis_2 = FDTD._normal_to_plane_basis(normal)
        for theta in np.linspace(-np.pi, np.pi, 1000):
            points = [
                self.world_to_grid_coords(
                    (
                        position
                        + radius * (np.cos(angle) * basis_1 + np.sin(angle) * basis_2)
                    )
                )
                for angle in [theta, theta + np.pi]
            ]
            line = skimage.draw.line_nd(*points)
            self.emitter_amps[line] = amp
            self.emitter_freqs[line] = freq
            self.emitter_phases[line] = phase
            self.emitter_angles[line] = torch.as_tensor(
                normal, dtype=self.emitter_angles.dtype, device=self.device
            )
            self.is_point_emitter[line] = False
        return self

    def world_to_grid_coords(self, position):
        return (
            (
                np.array(self.grid_dimensions)
                * np.array(position)
                / np.array(self.box_dimensions)
            )
            .round()
            .astype(int)
        )

    @staticmethod
    def _normal_to_plane_basis(normal):
        basis_1 = np.array([normal[0] + 1] + normal[1:].tolist())
        basis_1 -= (basis_1 @ normal) * normal
        basis_1 /= np.linalg.norm(basis_1)
        basis_2 = np.cross(normal, basis_1)
        basis_2 /= np.linalg.norm(basis_2)
        assert (
            np.isclose(basis_1 @ normal, 0.0)
            and np.isclose(basis_1 @ basis_2, 0.0)
            and np.isclose(basis_2 @ normal, 0.0)
        )
        return basis_1, basis_2
