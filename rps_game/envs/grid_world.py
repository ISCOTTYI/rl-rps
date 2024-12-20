import os
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


"""
Two pieces of the same type cannot be on top of each other irrespective of team.
Two pieces of opposing teams that can play RPS can be ontop of each other and the
winning piece annihilates the loosing piece. Each round one player moves one
piece.
"""

class Directions(Enum):
    # access via e.g. Directions.stay.value
    STAY = 0
    RIGHT = 1
    UP = 2
    LEFT = 3
    DOWN = 4


class RPSEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "console"],
        "render_fps": 4
    }

    def __init__(self, render_mode=None, size=5, n_pieces=18):
        # if render_mode != "console":
        #     raise NotImplementedError()
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.n_pieces = n_pieces
        self.n_directions = 5

        """
        Observation space:
        ------------------
        2 x size x size, first grid is player 0 second is player 1.
        Elements are 0-empty, 1-rock, 2-paper, 3-scissors
        """
        self.observation_space = spaces.Box(
            low=np.zeros((2, size, size)),
            high=np.full((2, size, size), 3),
            shape=(2, size, size), dtype=np.int8
        )

        self.state = None

        """
        Action space:
        -------------
        An action is one player moving one piece.
        Action passed as (x, y, direction), where direction = 0, 1, 2, 3, 4
        """
        self.action_space = spaces.Discrete(self.n_directions * size**2)
        self._dir_to_displace_vec = {
            Directions.STAY.value: np.array([0, 0]),
            Directions.RIGHT.value: np.array([0, 1]), # + one column
            Directions.UP.value: np.array([1, 0]), # + one row
            Directions.LEFT.value: np.array([0, -1]),
            Directions.DOWN.value: np.array([-1, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if render_mode == "human":
            self._load_render_assets()

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    
    def _move_to_action(self, move):
        # Converts move (e.g. (x=10, y=3, direc=UP)) to action in action_space
        return np.ravel_multi_index(move, (self.size, self.size, self.n_directions))
    
    def _action_to_move(self, action):
        # Converts action in action_space (e.g. 34) to move (x, y, direc)
        return np.unravel_index(action, (self.size, self.size, self.n_directions))

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return {
            "n_agent_pieces": self._count_pieces_by(0),
            "n_opponent_pieces": self._count_pieces_by(1)
        }
    
    def _count_pieces_by(self, player):
        return np.sum(self.state[player] != 0)

    def _sample_random_coords(self, seed=None):
        # Seed self.np_random
        super().reset(seed=seed)
        all_coords = [[x, y] for x in range(self.size) for y in range(self.size)]
        self.np_random.shuffle(all_coords)
        return all_coords[:self.n_pieces]
    
    def _init_state_random(self, seed=None):
        self.state = np.zeros((2, self.size, self.size), dtype=np.int8)
        coords = self._sample_random_coords(seed=seed)
        for i, (x, y) in enumerate(coords):
            self.state[i%2, x, y] = (i%3)+1

    def reset(self, seed=None, options=None):
        if options is None:
            self._init_state_random(seed=seed)
        else:
            raise NotImplementedError()
        observation = self._get_obs()
        info = self._get_info()
        self.render()
        return observation, info
    
    def move_valid(self, move):
        """
        Rules:
        ------
        Player 0 is controlled, only one piece per player in a position,
        two pieces of the same type cannot be on the same position (irrespective
        of team), don't move outside grid, only move pieces owned by player
        """
        # move is (x, y, dir), player 0 is moving
        x, y, dir = move
        piece_type = self.state[0, x, y]
        pos = np.array([x, y])
        displace_vec = self._dir_to_displace_vec[dir]
        new_pos = pos + displace_vec
        new_x, new_y = new_pos
        out_of_bounds = np.any((new_pos < 0) | (new_pos >= self.size))
        if out_of_bounds:
            return False
        occupied = bool(self.state[0, new_x, new_y]) # occupied by player 0
        not_fightable = (self.state[1, new_x, new_y] == piece_type) # enemy piece of same type on new_pos
        if occupied or not_fightable:
            return False
        return True
    
    def compute_action_mask(self):
        mask = np.zeros(self.n_directions * self.size**2, dtype=bool)
        # Construct list of indices of pieces of player 0
        piece_indices = np.argwhere(self.state[0] != 0)
        for x, y in piece_indices:
            piece_type = self.state[0, x, y]
            for dir in range(self.n_directions):
                move = (x, y, dir)
                if self.move_valid(move):
                    action = self._move_to_action((x, y, dir))
                    mask[action] = True
        return mask
    
    def _is_game_over(self):
        # Game over if either player has no pieces
        return not(self._count_pieces_by(0) and self._count_pieces_by(1))
    
    @staticmethod
    def handle_rps_fight(player_piece, opponent_piece):
        """
        Pieces are 0 - empty, 1 - rock, 2 - paper, 3 - scissors
        Returns:
            1 if player won, -1 if opponent won or 0 if tie or either was empty
        """
        p, o = player_piece, opponent_piece # alias
        return np.sign(p * o * ((p - o + 4) % 3 - 1))
    
    def _perform_move(self, move):
        x, y, dir = move
        piece_type = self.state[0, x, y]
        displace_vec = self._dir_to_displace_vec[dir]
        new_x, new_y = x + displace_vec[0], y + displace_vec[1]
        print(f"Moving a {piece_type} from {(x, y)} to {(new_x, new_y)} as per direction {dir} and displacement vec {displace_vec}")
        enemy = self.state[1, new_x, new_y] # could be empty
        outcome = RPSEnv.handle_rps_fight(piece_type, enemy)
        # TODO: overcomplicated, but kinda readable (?)
        if outcome > 0: # player 0 won
            self.state[0, x, y] = 0
            self.state[0, new_x, new_y] = piece_type
            self.state[1, new_x, new_y] = 0
        elif outcome == 0: # new cell was empty
            self.state[0, x, y] = 0
            self.state[0, new_x, new_y] = piece_type
        else: # opponent won
            self.state[0, x, y] = 0

    def step(self, action):
        """
        Action must be element of action_space.
        """
        mask = self.compute_action_mask()
        move = self._action_to_move(action) # (x, y, direction)
        if not mask[action]:
            raise ValueError(f"Invalid action {action} = {move}")
        self._perform_move(move)        
        terminated = self._is_game_over()

        # TODO!
        # reward = 1 if terminated else 0  # Binary sparse rewards
        reward = 0

        observation = self._get_obs()
        info = self._get_info()
        self.render()
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
        elif self.render_mode == "console":
            return self._render_console()
    
    def _render_console(self):
        print(self.state)
    
    def _load_render_assets(self):
        asset_base = "rps_game/envs/assets"
        cell_size = self.window_size // self.size
        self.images = {
            1: pygame.image.load(os.path.join(asset_base, "rock.png")),
            2: pygame.image.load(os.path.join(asset_base, "paper.png")),
            3: pygame.image.load(os.path.join(asset_base, "scissors.png"))
        }
        for key, image in self.images.items():
            self.images[key] = pygame.transform.scale(image, (cell_size // 2, cell_size // 2))
    
    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            # Increase window width for scoreboard
            self.window = pygame.display.set_mode((self.window_size + 150, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        colors = [(31, 119, 180), (214, 39, 40)]  # tab:blue and tab:red
        bg_color = (255, 255, 255)  # White background
        grid_color = (200, 200, 200)  # Light gray grid
        cell_size = self.window_size // self.size
        self.window.fill(bg_color)
        # Draw grid
        for x in range(self.size + 1):
            pygame.draw.line(self.window, grid_color, (x * cell_size, 0), (x * cell_size, self.window_size))
            pygame.draw.line(self.window, grid_color, (0, x * cell_size), (self.window_size, x * cell_size))
        # Draw pieces
        for player in range(2):  # Loop over players
            for x in range(self.size):
                for y in range(self.size):
                    piece = self.state[player, x, y]
                    if piece != 0:
                        piece_color = colors[player]
                        center = (y * cell_size + cell_size // 2, x * cell_size + cell_size // 2)
                        pygame.draw.circle(self.window, piece_color, center, cell_size // 3)

                        # Render the image at the piece's position
                        image = self.images[piece]
                        image_rect = image.get_rect(center=center)
                        self.window.blit(image, image_rect)
        # Draw scoreboard
        scoreboard_x = self.window_size + 10
        font = pygame.font.SysFont(None, 24)
        for player in range(2):
            player_color = colors[player]
            player_text = font.render(f"Player {player}:", True, player_color)
            self.window.blit(player_text, (scoreboard_x, 20 + player * 100))
            # Count pieces
            counts = {1: 0, 2: 0, 3: 0}
            for x in range(self.size):
                for y in range(self.size):
                    piece = self.state[player, x, y]
                    if piece in counts:
                        counts[piece] += 1
            # Render counts
            for i, (piece, count) in enumerate(counts.items()):
                text = font.render(f"{['Rock', 'Paper', 'Scissors'][piece - 1]}: {count}", True, player_color)
                self.window.blit(text, (scoreboard_x, 40 + player * 100 + i * 20))
        # Update display and manage frame rate
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        # Handle events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
