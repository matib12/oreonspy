''' ### CAVITY ENVIRONMENT ### '''

import math
from typing import Optional, Tuple, Union

import numpy as np
from scipy import constants as const

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
#from gymnasium.vector import VectorEnv
#from gymnasium.vector.utils import batch_space

import oreonspy as op


class CavityEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None, t_a = 0.1, r_a = 0.9, r_b = 0.9, L=3000., f_c = 50.e3, E_in = 1, lambd = 1064e-9):
    
    #   self.gravity = 9.8

    #   self.mass_mirror1 = 1.0 # mass of one mirror
    #   self.mass_mirror2 = 1.0
    #   self.r_1 = 0.9 # reflectivity of one mirror
    #   self.r_2 = 0.9
    #   self.t_1 = 0.1 # transmittivity of one mirror
    #   self.t_2 = 0.1
    #   self.T = self.L  / const.c
    #   self.L = 3000.  # Cavity lenght
    #   self.total_mass = self.masspole + self.masscart
    #   self.pole_length = self.masspole * self.length
    #   self.force_mag = 10.0
    #   self.tau = 1 / self.f_c  # seconds between state updates 

    #   self.kinematics_integrator = "euler"   
        
        #self.P = 0  # Power of the cavity field
        #self.d_zeta = d_zeta

        # LASER
        self.E_in = E_in  #
        self.lambd = lambd  # m
        self.k = 2.*np.pi / self.lambd

        self.cavity = op.Cavity(t_a , r_a , r_b , L) 
        epsilon = np.random.uniform(-0.05*self.lambd,0.05*self.lambd)
        self.cavity.L += epsilon

        # SAMPLING 
        self.f_c = f_c # Calculation frequency
        self.tau = 1 / self.f_c  # seconds between state updates


    
    #   Lenght at which to fail the episode
        self.lengh_limit = self.cavity.L + 3*self.lambd

        self.action_space = spaces.Box(-3*self.lambd, 3*self.lambd, dtype=np.float32)
        self.observation_space = spaces.Box(0., 1., dtype=np.float32)

        #self.render_mode = render_mode

        #self.screen_width = 600
        #self.screen_height = 400
        #self.screen = None
        #self.clock = None
        #self.isopen = True
        
        self.state = None
        self.steps_beyond_terminated = None

    def step(self, d_zeta):
        assert self.action_space.contains(
            
        ), f"{d_zeta!r} ({type(d_zeta)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        #shift = self.d_zeta if action == 1 else -self.d_zeta

        P, self.cavity.L = self.state
        #d_zeta = np.random.uniform(0, 0.05 * self.lambd)
        self.cavity.simulation(self.k, self.f_c)
        P = np.abs(self.cavity.sim_step(d_zeta, self.E_in))**2
        self.cavity.L += d_zeta
        self.state = (P, self.cavity.L)

        #n = int( np.random.uniform(0,5))
        terminated = bool(
            0.05 <= P <= 0.25  or P >=0.5
            or self.cavity.L > self.lengh_limit
            or self.cavity.L == np.ceil(self.cavity.L/(self.lambd/2))*self.lambd
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        #truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        
        P = self.np.random.uniform(0., 0.005)
        epsilon = np.random.uniform(-0.05*self.lambd,0.05*self.lambd)
        self.cavity.L += epsilon
        self.steps_beyond_terminated = None
        self.state = (P, self.cavity.L)

        '''if self.render_mode == "human":
            self.render()
        '''
        return np.array(self.state, dtype=np.float32), {}

    '''def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False '''



