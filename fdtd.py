import numpy as np
import jax
import jax.numpy as jnp
import skimage.draw
from tqdm import tqdm


def _fdtd_step(
    pressure,
    vel_x,
    vel_y,
    vel_z,
    vel_coef,
    pressure_coef,
    solid_damping,
    emitter_amps,
    emitter_freqs,
    emitter_phases,
    emitter_angles,
    is_point_emitter,
    t,
    attenuation_dt,
    base_pressure,
):
    new_vel_x = solid_damping * (
        vel_x - vel_coef * (-pressure).at[:-1, :, :].add(pressure[1:, :, :])
    )
    new_vel_y = solid_damping * (
        vel_y - vel_coef * (-pressure).at[:, :-1, :].add(pressure[:, 1:, :])
    )
    new_vel_z = solid_damping * (
        vel_z - vel_coef * (-pressure).at[:, :, :-1].add(pressure[:, :, 1:])
    )

    phase = jnp.sin(2 * jnp.pi * emitter_freqs * t - emitter_phases)

    directed_emitter_amps = jnp.where(
        jnp.logical_not(is_point_emitter), emitter_amps, 0.0
    )

    new_vel_x = (
        new_vel_x * (1 - directed_emitter_amps)
        + directed_emitter_amps * emitter_angles[..., 0] * phase
    )
    new_vel_y = (
        new_vel_y * (1 - directed_emitter_amps)
        + directed_emitter_amps * emitter_angles[..., 1] * phase
    )
    new_vel_z = (
        new_vel_z * (1 - directed_emitter_amps)
        + directed_emitter_amps * emitter_angles[..., 2] * phase
    )

    new_pressure = (
        pressure
        - pressure_coef
        * (
            new_vel_x.at[1:, :, :].add(-new_vel_x[:-1, :, :])
            + new_vel_y.at[:, 1:, :].add(-new_vel_y[:, :-1, :])
            + new_vel_z.at[:, :, 1:].add(-new_vel_z[:, :, :-1])
        )
    ) / attenuation_dt

    point_emitter_amps = jnp.where(is_point_emitter, emitter_amps, 0.0)
    new_pressure = (
        jnp.where(jnp.logical_not(is_point_emitter), new_pressure, base_pressure)
        + phase * point_emitter_amps
    )

    return new_pressure, new_vel_x, new_vel_y, new_vel_z


class FDTD:
    C = 343
    AIR_DENSITY = 1.225
    BASE_PRESSURE = 0
    DAMPING_COEF = 0.999

    def __init__(
        self,
        box_dimensions,
        ds,
        solid,
        cfl_factor=0.5,
        pml_layers=8,
        dims_include_pml=False,
        dtype=jnp.float32,
    ):
        self.ds = ds
        self.dt = cfl_factor * self.ds / (jnp.sqrt(2) * self.C)
        self.pressure_coef = self.AIR_DENSITY * self.C**2 * self.dt / self.ds
        self.vel_coef = self.dt / (self.ds * self.AIR_DENSITY)
        self.dtype = dtype

        self.box_dimensions = (
            box_dimensions
            if dims_include_pml
            else [dim + (2 * pml_layers) * ds for dim in box_dimensions]
        )
        self.grid_dimensions = (
            (
                (0 if dims_include_pml else 2 * pml_layers)
                + jnp.round(jnp.asarray(box_dimensions) / self.ds)
            )
            .astype(int)
            .tolist()
        )

        self.pressure = jnp.empty(self.grid_dimensions, dtype=dtype)
        self.vel_x = jnp.empty(self.grid_dimensions, dtype=dtype)
        self.vel_y = jnp.empty(self.grid_dimensions, dtype=dtype)
        self.vel_z = jnp.empty(self.grid_dimensions, dtype=dtype)
        self.attenuation = jnp.ones(self.grid_dimensions, dtype=dtype)
        self.amplitude = jnp.zeros(self.grid_dimensions, dtype=dtype)
        self._make_pml(pml_layers)
        self.solid = solid
        self.emitter_amps = jnp.zeros(self.grid_dimensions, dtype=dtype)
        self.emitter_freqs = jnp.zeros(self.grid_dimensions, dtype=dtype)
        self.emitter_phases = jnp.zeros(self.grid_dimensions, dtype=dtype)
        self.emitter_angles = jnp.zeros(self.grid_dimensions + [3], dtype=dtype)
        self.is_point_emitter = jnp.zeros(self.grid_dimensions, dtype=bool)
        self.fdtd_step = jax.jit(_fdtd_step)
        self.reset()

    def _make_pml(self, pml_layers):
        sigma_max = 0.5 / self.dt
        for i in range(pml_layers):
            j = None if i == 0 else -i
            attenuation = sigma_max * (1 - i / pml_layers)
            self.attenuation = (
                self.attenuation.at[i, i:j, i:j]
                .set(attenuation)
                .at[-1 - i, i:j, i:j]
                .set(attenuation)
                .at[i:j, i, i:j]
                .set(attenuation)
                .at[i:j, -1 - i, i:j]
                .set(attenuation)
                .at[i:j, i:j, i]
                .set(attenuation)
                .at[i:j, i:j, -1 - i]
                .set(attenuation)
            )

    def reset(self):
        self.pressure = self.pressure.at[...].set(self.BASE_PRESSURE)
        self.vel_x = self.vel_x.at[...].set(0)
        self.vel_y = self.vel_y.at[...].set(0)
        self.vel_z = self.vel_z.at[...].set(0)
        self.solid_damping = (
            (1 - self.solid) * self.DAMPING_COEF / (self.attenuation * self.dt + 1)
        )
        self.amplitude = self.amplitude.at[...].set(0)
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
            self.pressure, self.vel_x, self.vel_y, self.vel_z = self.fdtd_step(
                self.pressure,
                self.vel_x,
                self.vel_y,
                self.vel_z,
                self.vel_coef,
                self.pressure_coef,
                self.solid_damping,
                self.emitter_amps,
                self.emitter_freqs,
                self.emitter_phases,
                self.emitter_angles,
                self.is_point_emitter,
                self.t,
                attenuation_dt,
                self.BASE_PRESSURE,
            )

            if amp_measurement_warmup is not None and i >= amp_measurement_warmup:
                self._get_amp()

            self.t += self.dt

    def _get_amp(self):
        self.amplitude = jnp.maximum(self.amplitude, jnp.abs(self.pressure))

    def _assert_can_add_emitter(self):
        assert (
            self.t == 0
        ), "Cannot add an emitter to a started simulation. Re-instantiate the object or call reset()"

    def add_point_emitter(self, position, amp, freq, phase):
        self._assert_can_add_emitter()
        position = self.world_to_grid_coords(position)
        self.emitter_amps = self.emitter_amps.at[*position].set(amp)
        self.emitter_freqs = self.emitter_freqs.at[*position].set(freq)
        self.emitter_phase = self.emitter_phases.at[*position].set(phase)
        self.is_point_emitter = self.is_point_emitter.at[*position].set(True)
        return self

    def add_circular_emitter(self, position, amp, freq, phase, angle, radius):
        self._assert_can_add_emitter()
        normal = jnp.asarray(
            [
                jnp.sin(angle[0]),
                jnp.cos(angle[0]) * jnp.cos(jnp.pi / 2 + angle[1]),
                jnp.cos(angle[0]) * jnp.cos(angle[1]),
            ]
        )
        normal /= jnp.linalg.norm(normal)
        basis_1, basis_2 = FDTD._normal_to_plane_basis(normal)
        angles = jnp.linspace(-jnp.pi, jnp.pi, 1000).reshape(-1, 1)
        points = jnp.hstack(
            [
                self.world_to_grid_coords(
                    (
                        jnp.asarray(position)
                        + radius * (jnp.cos(theta) * basis_1 + jnp.sin(theta) * basis_2)
                    )
                )
                for theta in [angles, angles + jnp.pi]
            ]
        )
        points = jnp.unique(points, axis=0)
        circle_indices = ([], [], [])
        for point_pair in points:
            line = skimage.draw.line_nd(point_pair[:3], point_pair[3:])
            for i, line_axis in enumerate(line):
                circle_indices[i].append(line_axis)
        circle_indices = tuple([np.concatenate(axis) for axis in circle_indices])
        self.emitter_amps = self.emitter_amps.at[circle_indices].set(amp)
        self.emitter_freqs = self.emitter_freqs.at[circle_indices].set(freq)
        self.emitter_phases = self.emitter_phases.at[circle_indices].set(phase)
        self.emitter_angles = self.emitter_angles.at[circle_indices].set(normal)
        self.is_point_emitter = self.is_point_emitter.at[circle_indices].set(False)
        return self

    def world_to_grid_coords(self, position):
        position = jnp.asarray(position) + jnp.asarray(self.box_dimensions) / 2
        return (
            (
                jnp.asarray(self.grid_dimensions)
                * position
                / jnp.asarray(self.box_dimensions)
            )
            .round()
            .astype(int)
        )

    @staticmethod
    def _normal_to_plane_basis(normal, atol=3e-8):
        vectors = [
            jnp.asarray([0, normal[2], -normal[1]]),
            jnp.asarray([-normal[2], 0, normal[0]]),
            jnp.asarray([normal[1], -normal[0], 0]),
        ]
        basis = []
        for vector in vectors:
            if not jnp.allclose(vector, 0.0):
                basis.append(vector / jnp.linalg.norm(vector))
            if len(basis) == 2:
                break
        assert (
            jnp.isclose(basis[0] @ normal, 0.0, atol=atol)
            and jnp.isclose(basis[0] @ basis[1], 0.0, atol=atol)
            and jnp.isclose(basis[1] @ normal, 0.0, atol=atol)
        )
        return basis[0], basis[1]
