import numpy as np
import tensorflow as tf

from datasets.utils.clock_signal import clock_signal
from module.scorbot import Scorbot
from utils.plots import plot_coordinates


class BrownianMotion:
    def __init__(self, batch_size, dims, length, dt, delta, seed):

        self.batch_size = batch_size
        self.dims = dims
        self.length = length
        self.dt = dt
        self.delta = delta
        self.seed = seed
        self.x0 = 0.0

    def gen_random_walk(self):
        w = tf.ones((self.length, self.batch_size, self.dims)) * self.x0

        def brownian(m, x):  # noqa
            yi = tf.random.normal((self.batch_size, self.dims), seed=self.seed,
                                  mean=0.0, stddev=self.delta * tf.math.sqrt(tf.cast(self.dt, tf.float32)))
            return m + yi

        return tf.transpose(tf.scan(brownian, w, initializer=w[0]), perm=[1, 0, 2])


class BrownianTrajectories:
    def __init__(self, batch_size, trajectory_length, clock_input_size, coordinate_input_size, num_joints=2,
                 joint_length=0.5, lsg_timesteps_per_trainee_timestep=1, window_length=50, step_size=0.01, seed=42):

        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        self.clock_input_size = clock_input_size
        self.coordinate_input_size = coordinate_input_size
        self.num_joints = num_joints
        self.joint_length = joint_length
        self.lsg_timesteps_per_trainee_timestep = lsg_timesteps_per_trainee_timestep
        self.window_length = window_length
        self.step_size = step_size
        self.seed = seed
        self.start = tf.tile(tf.expand_dims(tf.constant(np.array([0., np.pi/2], np.float32)), axis=0), [batch_size, 1])

        self.noise = BrownianMotion(batch_size, num_joints, trajectory_length, step_size, 1.0, seed)

    def generate_data(self):
        angular_velocities = self.noise.gen_random_walk()

        padded_angular_vel = tf.concat((tf.zeros_like(
          tf.reverse(angular_velocities[:, 0:self.window_length - 1], axis=[1])), angular_velocities,
                                        tf.reverse(angular_velocities[:, -self.window_length + 1:], axis=[1])), axis=1)
        window = tf.signal.hann_window(self.window_length)

        window = window / tf.reduce_sum(window)

        angular_velocities = tf.nn.convolution(
            padded_angular_vel,
            tf.tile(
              tf.expand_dims(
                tf.expand_dims(window, axis=-1), axis=-1), [1, 1, padded_angular_vel.shape[-1]]),
            padding='VALID')[:, int(self.window_length // 2):(int(self.window_length // 2) + self.trajectory_length)]

        angular_velocities_scaled = tf.clip_by_value(angular_velocities, -np.pi, np.pi)

        angles_scaled = tf.expand_dims(self.start, axis=1) \
            + self.step_size * tf.math.cumsum(angular_velocities_scaled, axis=1)

        cartesian_x_scaled = \
            tf.expand_dims((tf.cos(angles_scaled[:, :, 0]) + tf.cos(angles_scaled[:, :, 1] + angles_scaled[:, :, 0]))
                           * self.joint_length, axis=-1)
        cartesian_y_scaled = \
            tf.expand_dims((tf.sin(angles_scaled[:, :, 0]) + tf.sin(angles_scaled[:, :, 1] + angles_scaled[:, :, 0]))
                           * self.joint_length, axis=-1)

        cartesian_scaled = tf.concat((cartesian_x_scaled, cartesian_y_scaled), axis=-1)

        return cartesian_scaled, angular_velocities_scaled, angles_scaled

    def get_data(self):
        coordinates, angular_velocities, angles = self.generate_data()

        x_b = np.cast['int64']((coordinates[:, :, 0] + 1) * 2**(self.coordinate_input_size - 1))
        x_b = ((x_b[:, :, None] & (1 << np.arange(self.coordinate_input_size))) > 0).astype(int)

        y_b = np.cast['int64']((coordinates[:, :, 1] + 1) * 2**(self.coordinate_input_size - 1))
        y_b = ((y_b[:, :, None] & (1 << np.arange(self.coordinate_input_size))) > 0).astype(int)

        coordinates_encoded = tf.concat((tf.constant(x_b), tf.constant(y_b)), -1)
        coordinates_encoded_padded = tf.repeat(coordinates_encoded,
                                               np.full(coordinates_encoded.shape[1],
                                                       self.lsg_timesteps_per_trainee_timestep), axis=1)

        mask = np.zeros((1, 1, self.lsg_timesteps_per_trainee_timestep, 1))
        mask[0][0][self.lsg_timesteps_per_trainee_timestep - 1] = 1
        mask = tf.reshape(tf.tile(tf.constant(mask), [1, self.trajectory_length, 1, 1]),
                          (1, self.trajectory_length * self.lsg_timesteps_per_trainee_timestep, 1))

        return tf.cast(coordinates_encoded_padded, tf.float32), tf.cast(coordinates, tf.float32), \
            tf.cast(angular_velocities, tf.float32), tf.cast(angles, tf.float32), tf.cast(mask, tf.bool)

    def get_static_data(self):

        clock_signal_lsg = tf.constant(
            clock_signal(
                self.trajectory_length * self.lsg_timesteps_per_trainee_timestep,
                self.clock_input_size,
                self.clock_input_size * self.lsg_timesteps_per_trainee_timestep,
                1,
                10
            )
        )
        clock_signal_lsg_batched = tf.tile(tf.expand_dims(clock_signal_lsg, axis=0), [self.batch_size, 1, 1])

        clock_signal_trainee = tf.constant(
            clock_signal(
                self.trajectory_length,
                self.clock_input_size,
                self.clock_input_size,
                1,
                10
            )
        )
        clock_signal_trainee_batched = tf.tile(tf.expand_dims(clock_signal_trainee, axis=0), [self.batch_size, 1, 1])

        init_position = self.start

        return tf.cast(init_position, tf.float32), tf.cast(clock_signal_trainee_batched, tf.float32), \
            tf.cast(clock_signal_lsg_batched, tf.float32)


class BrownianTrajectoriesScorbot:
    def __init__(self, batch_size, trajectory_length, clock_input_size, coordinate_input_size,
                 num_joints=Scorbot.num_joints, joints=(0, 1, 2, 3), lsg_timesteps_per_trainee_timestep=1,
                 window_length=50, step_size=0.01, seed=42):
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        self.clock_input_size = clock_input_size
        self.coordinate_input_size = coordinate_input_size
        self.num_joints = num_joints
        self.joints = joints
        self.lsg_timesteps_per_trainee_timestep = lsg_timesteps_per_trainee_timestep
        self.window_length = window_length
        self.step_size = step_size
        self.seed = seed
        self.start = tf.tile(tf.expand_dims(tf.constant(np.array([0.0] * num_joints, np.float32)), axis=0),
                             [batch_size, 1])

        self.scorbot = Scorbot()
        self.noise = BrownianMotion(batch_size, num_joints, trajectory_length, step_size, 0.65, seed)

    def generate_data(self):
        angular_velocities = self.noise.gen_random_walk()
        mask = np.zeros_like(angular_velocities)
        for i in range(mask.shape[-1]):
            if i in self.joints:
                mask[:, :, i] = 1
        angular_velocities = mask * angular_velocities

        padded_angular_vel = tf.concat((tf.zeros_like(
          tf.reverse(angular_velocities[:, 0:self.window_length - 1], axis=[1])), angular_velocities,
                                        tf.reverse(angular_velocities[:, -self.window_length + 1:], axis=[1])), axis=1)
        window = tf.signal.hann_window(self.window_length)

        window = window / tf.reduce_sum(window)

        angular_velocities = tf.nn.convolution(
            padded_angular_vel,
            tf.tile(
              tf.expand_dims(
                tf.expand_dims(window, axis=-1), axis=-1), [1, 1, padded_angular_vel.shape[-1]]),
            padding='VALID')[:, int(self.window_length // 2):(int(self.window_length // 2) + self.trajectory_length)]

        angular_velocities_scaled = tf.clip_by_value(angular_velocities, -np.pi, np.pi)

        angles_scaled = tf.expand_dims(self.start, axis=1) + \
            self.step_size * tf.math.cumsum(angular_velocities_scaled, axis=1)

        clip_value_min = [Scorbot.save_space[j][0] * np.pi/180 for j in self.joints]
        clip_value_max = [Scorbot.save_space[j][1] * np.pi / 180 for j in self.joints]
        angles_scaled = tf.clip_by_value(angles_scaled, clip_value_min=clip_value_min, clip_value_max=clip_value_max)

        # TODO does not work in the general case
        args = {'q{0}'.format(joint+1): angles_scaled[:, :, joint] for joint in self.joints}
        cartesian_scaled = self.scorbot.direct_kinematics(**args)

        cartesian_x_scaled = tf.expand_dims(cartesian_scaled[0], axis=-1)
        cartesian_y_scaled = tf.expand_dims(cartesian_scaled[1], axis=-1)
        cartesian_z_scaled = tf.expand_dims(cartesian_scaled[2], axis=-1)

        cartesian_scaled = tf.concat((cartesian_x_scaled, cartesian_y_scaled, cartesian_z_scaled), axis=-1)

        return cartesian_scaled, angular_velocities_scaled, angles_scaled

    def get_data(self):
        coordinates, angular_velocities, angles = self.generate_data()

        x_b = np.cast['int64']((coordinates[:, :, 0] + 1) * 2**(self.coordinate_input_size - 1))
        x_b = ((x_b[:, :, None] & (1 << np.arange(self.coordinate_input_size))) > 0).astype(int)

        y_b = np.cast['int64']((coordinates[:, :, 1] + 1) * 2**(self.coordinate_input_size - 1))
        y_b = ((y_b[:, :, None] & (1 << np.arange(self.coordinate_input_size))) > 0).astype(int)

        z_b = np.cast['int64']((coordinates[:, :, 2] + 1) * 2**(self.coordinate_input_size - 1))
        z_b = ((z_b[:, :, None] & (1 << np.arange(self.coordinate_input_size))) > 0).astype(int)

        coordinates_encoded = tf.concat((tf.constant(x_b), tf.constant(y_b), tf.constant(z_b)), -1)
        coordinates_encoded_padded = tf.repeat(coordinates_encoded,
                                               np.full(coordinates_encoded.shape[1],
                                                       self.lsg_timesteps_per_trainee_timestep), axis=1)

        mask = np.zeros((1, 1, self.lsg_timesteps_per_trainee_timestep, 1))
        mask[0][0][self.lsg_timesteps_per_trainee_timestep - 1] = 1
        mask = tf.reshape(tf.tile(tf.constant(mask), [1, self.trajectory_length, 1, 1]),
                          (1, self.trajectory_length * self.lsg_timesteps_per_trainee_timestep, 1))
        print('')
        return tf.cast(coordinates_encoded_padded, tf.float32), tf.cast(coordinates, tf.float32), \
            tf.cast(angular_velocities, tf.float32), tf.cast(angles, tf.float32), tf.cast(mask, tf.bool)

    def get_static_data(self):

        clock_signal_lsg = tf.constant(
            clock_signal(
                self.trajectory_length * self.lsg_timesteps_per_trainee_timestep,
                self.clock_input_size,
                self.clock_input_size * self.lsg_timesteps_per_trainee_timestep,
                1,
                10
            )
        )
        clock_signal_lsg_batched = tf.tile(tf.expand_dims(clock_signal_lsg, axis=0), [self.batch_size, 1, 1])

        clock_signal_trainee = tf.constant(
            clock_signal(
                self.trajectory_length,
                self.clock_input_size,
                self.clock_input_size,
                1,
                10
            )
        )
        clock_signal_trainee_batched = tf.tile(tf.expand_dims(clock_signal_trainee, axis=0), [self.batch_size, 1, 1])

        init_position = self.start

        return tf.cast(init_position, tf.float32), tf.cast(clock_signal_trainee_batched, tf.float32), \
            tf.cast(clock_signal_lsg_batched, tf.float32)


# trajectories = BrownianTrajectoriesScorbot(batch_size=90, trajectory_length=250, clock_input_size=5,
#                                            coordinate_input_size=16, num_joints=2, joints=[0, 1], seed=95)
# _, cartesian, angular_velocities, angles, _ = trajectories.get_data()
# angles = angles * 180/np.pi
#
# print(np.min(angles, axis=(0, 1)), np.max(angles, axis=(0, 1)))
# coordinate_fig = plot_coordinates(
#     outputs={'train': angles, 'test': angles,  # are either angles or angular velocities
#              'cartesian_train': cartesian, 'cartesian_test': cartesian},
#     targets={'omega': angular_velocities, 'angles': angles, 'cartesian': cartesian},
#     model_outputs_angles=True,
#     num_timesteps=250)
