from cProfile import label

from gym_classics2.envs.abstract.base_env import BaseEnv
from gymnasium.spaces import MultiDiscrete

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors


def _load_pygame():
    try:
        import pygame
    except ImportError as error:
        raise ImportError(
            "Rendering requires pygame; install gym-classics2 with the "
            "'render' extra."
        ) from error
    return pygame

# we need to overwrite:
# * _next_state
# * _reward
# * _done
# * _generate_transitions

class Gridworld(BaseEnv):
    """Finite rectangular gridworld constructed from an ASCII layout.

    Layout rows are written from top to bottom; raw states are ``(x, y)``
    coordinates with ``(0, 0)`` in the lower-left corner. ``S`` marks a start,
    ``G`` a terminal goal, ``X`` a blocked cell, and a space a traversable cell.
    Other characters are traversable labels retained for plotting. Optional ``|``
    characters are ignored when the layout is parsed.

    Actions and transitions:

    The default actions move up, right, down, and left and are represented in that 
    order by the integers 0 to 3. The default transition model is deterministic.
    An action that would leave the gridworld or enter 
    a blocked cell leaves the agent in place. Additional actions can be added in the constructor and will use 
    integer IDs starting at 4. The transition model can be changed
    by subclassing and overwriting ``_next_state``. 
    
    For an example of a stochastic transition model, see :class:`ClassicGridworld`
    
    Reward model:
    
    Entering a goal yields ``goal_reward`` and terminates the episode; other transitions yield
    ``step_reward``. Since goal states are terminal (absorbing), the reward for a transition from 
    a goal state is always zero.
    
    The default setting is for a world without step cost and reaching the goal is rewarded with 1.0. If you 
    want to use a step cost, set ``step_reward`` to a negative value. Note that the ``goal_reward`` 
    is for the transition to the goal state and thus needs to be 
    adjusted to reflect the positive reward for reaching the goal minus the cost of getting there.
    
    You can also overwrite the ``_reward`` method to implement a custom reward model.

    Args:
        layout_string: Rectangular ASCII representation of the grid.
        action_labels: Labels for the four actions, ordered as up, right, down,
            and left.
        goal_reward: Reward for a transition into a goal cell.
        step_reward: Reward for any other transition.
        tabular: If true, observations are integer state IDs. If false, they are
            raw ``(x, y)`` coordinates.
        render_mode: ``None`` to disable rendering, ``"human"`` for a window, or
            ``"rgb_array"`` for an RGB image returned by ``render``.

    Note:
        Rendering requires the optional ``render`` dependency extra.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, layout_string, action_labels = ["up", "right", "down", "left"], 
                 goal_reward = 1.0, step_reward = 0.0, 
                 tabular = True, render_mode=None):
        """Initialize the gridworld from ``layout_string``."""
        
        self._goal_reward = goal_reward
        self._step_reward = step_reward
        
        self.dims, starts, self._goals, self._blocks, self._extra_labels = parse_gridworld(layout_string)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.PyGame_window_size = 512  # The size of the PyGame window in pixels
        self.PyGame_window = None
        self.PyGame_clock = None

        super().__init__(starts, action_labels = action_labels, tabular = tabular, reachable_states = None)
        
        if not tabular:
            self.observation_space = MultiDiscrete(self.dims)

    @property
    def _next_state(self, state, action, *random_elements):
        next_state = self._move(state, action)
        if self._is_blocked(next_state):
            next_state = state
        return self._clamp(next_state), 1.0
    
    def _generate_transitions(self, state, action):
        yield self._deterministic_step(state, action)
    
    def _reward(self, state, action, next_state):      
        if state in self._goals:
            return 0.0
        if next_state in self._goals: 
            return self._goal_reward 
        return self._step_reward
    
    def _done(self, state, action, next_state):
        return next_state in self._goals
    
    def step(self, action):
        """Advance the environment and render a frame in human mode."""
        next_state, reward, done, x, info = super().step(action)
        
        if self.render_mode == "human":
            self._render_frame()
        
        return next_state, reward, done, x, info

    def reset(self, seed=None, options=None):
        """Reset to a start cell and render a frame in human mode."""
        observation, info = super().reset(seed=seed, options=options)
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def _move(self, state, action):
        x, y = state
        return {
            0: (x,   y+1),  # Up
            1: (x+1, y),    # Right
            2: (x,   y-1),  # Down
            3: (x-1, y)     # Left
        }[action]

    def _clamp(self, state):
        """Clamps the state within the grid dimensions."""
        x, y = state
        x = max(0, min(x, self.dims[0] - 1))
        y = max(0, min(y, self.dims[1] - 1))
        return (x, y)

    def goal_states(self):
        """Tuple containing the raw terminal goal coordinates."""
        return tuple(sorted(self._goals))

    def _is_blocked(self, state):
        """Returns True if this state cannot be occupied, False otherwise."""
        return state in self._blocks
    
    def close(self):
        """Release any PyGame display resources."""
        if self.PyGame_window is not None:
            pygame = _load_pygame()
            pygame.display.quit()
            pygame.quit()
            self.PyGame_window = None
            self.PyGame_clock = None
        
        super().close()
    
    def render(self):
        """Return an RGB frame when ``render_mode='rgb_array'``."""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        pygame = _load_pygame()
        pix_square_size = self.PyGame_window_size / max(self.dims)
        display_size = np.array(pix_square_size) * self.dims
        
        if self.PyGame_window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.PyGame_window = pygame.display.set_mode(display_size)
        if self.PyGame_clock is None and self.render_mode == "human":
            self.PyGame_clock = pygame.time.Clock()

        canvas = pygame.Surface(display_size)
        canvas.fill((255, 255, 255))

        # draw the goal
        for g in self._goals:
            pos = np.array(g)
            pos[1] = self.dims[1] - pos[1] - 1
            pygame.draw.rect(
                canvas,
                (0, 128, 0),
                pygame.Rect(
                    pix_square_size * pos,
                    (pix_square_size, pix_square_size),
                ),
            )
     
        # unreachable squares
        for g in np.argwhere(self.to_matrix() == -1):
            pos = np.array(g)
            pos = np.flip(pos)
            pos[1] = self.dims[1] - pos[1] - 1
            pygame.draw.rect(
                canvas,
                (72, 72, 72),
                pygame.Rect(
                    pix_square_size * pos,
                    (pix_square_size, pix_square_size),
                ),
            )
        
        # draw the agent
        pos = np.array(self.state)
        pos[1] = self.dims[1] - pos[1] - 1
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (pos + .5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.dims[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (display_size[0], pix_square_size * x),
                width=3,
            )
            
        for x in range(self.dims[0] + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, display_size[1]),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.PyGame_window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.PyGame_clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    ### Addition to the interface
    
    ## overwrite so we have arrows
    def id2action(self, action, type="text"):
        """Converts a numeric action ID into a label. Choices for type are 'text' and 'arrow'."""
        action = int(action)
        
        action_labels = self.action_labels + [""]  # Add empty label for hidden actions
       
        # empty has index 4 and is used to hide actions
        if (type == "arrow"):
             action_labels[0:4] = ['↑', '→', '↓', '←']
            
        return action_labels[action]

    def to_matrix(self, value = None):
        """Arrange state-indexed values as a grid-shaped matrix.

        Args:
            value: One value per reachable state. If omitted, use state IDs.

        Returns:
            Matrix with grid rows ordered for display. Unreachable cells contain
            ``-1``, an empty string, ``NaN``, or zero according to the input type.
        """
        
        if value is None:
            value = list(self.states())
        
        value = np.array(value)
      
        if np.issubdtype(value.dtype, np.integer):
            m = np.full(self.dims, -1, dtype=value.dtype)
        elif np.issubdtype(value.dtype, np.str_):
            m = np.full(self.dims, "", dtype=value.dtype)
        elif np.issubdtype(value.dtype, np.floating):
            m = np.full(self.dims, np.nan, dtype=value.dtype)
        else:
            m = np.zeros(self.dims, dtype=value.dtype)

        for y in range(self.dims[1]):
            for x in range(self.dims[0]):
                state = (x, y)
                if self.is_reachable(state):
                    m[x,y] = value[self.state2id(state)]
                else:
                    pass

        return m.transpose() 
    
    def print(self, array, decimals=2, separator=' ' * 2, signed=True, transpose=False):
        """Prints a gridworld array in a human-readable format. The array should be a vector with values for states in the gridworld, 
        such as a value function or policy."""
        
        def formatter(x):
            string = '{:' + ('+' if signed else '') + '.' + str(decimals) + 'f}'
            return string.format(x)
        maxlen = max([len(formatter(x)) for x in array])

        # Now we can actually print the values
        for y in reversed(range(self.dims[1])):
            for x in range(self.dims[0]):
                state = (x, y) if not transpose else (y, x)
                if self.is_reachable(state):
                    s = self.state2id(state)
                    print(formatter(array[s]).rjust(maxlen), end=separator)
                else:
                    print(' ' * maxlen, end=separator)
            print()
    
    def image(self, V=None, policy=None, episode = None, labels=None, title=None, cmap = 'auto', origin='lower', clim = None):
        """Display the gridworld, values, and optional actions as an image.

        Args:
            V: One value per state, such as a value function. If omitted, cells
                display their state IDs and the color bar is hidden.
            policy: One action ID per state. Actions are drawn as arrows and
                replace ``labels`` when provided.
            episode: Sequence of transitions whose first two entries are the state
                and action IDs. Actions taken in visited states are drawn as arrows
                and replace ``policy``.
            labels: One label per state. If ``True``, use values from ``V`` rounded
                to two decimal places.
            title: Plot title.
            cmap: Matplotlib colormap or colormap name. ``"auto"`` selects a
                sequential or diverging colormap from the displayed values.
            origin: ``"lower"`` places coordinate ``(0, 0)`` at the lower-left;
                ``"upper"`` places it at the upper-left.
            clim: Optional ``(minimum, maximum)`` limits for the color scale.
        """
        
        colorbar = True
        
        if not V is None:
            m = self.to_matrix(V)
        else:
            m = np.zeros(self.dims).transpose()
            # missing positions have -1
            m[self.to_matrix(labels) == -1] = np.nan
            labels = self.states()
            colorbar = False

        if not episode is None:
            # start with policy that hides all actions with an index past the last action.
            policy = np.full(len(self.states()), len(self.actions()))
            for step in episode:
                policy[step[0]] = step[1]

        if not policy is None:
            labels = [self.id2action(a, type = "arrow") for a in policy]

        if isinstance(labels, bool) and labels:
                labels = np.round(V, 2)

        if not labels is None:
            labels = self.to_matrix(labels)
            
        extra = [""] * self.dims[0] * self.dims[1]
        for s in self._starts:
            extra[self.state2id(s)] = "S"
        for s in self._goals:
            extra[self.state2id(s)] = "G"
        for s, label in self._extra_labels:
            extra[self.state2id(s)] = label
        extra = self.to_matrix(extra)
        
        _image(m, title=title, labels=labels, extra=extra, cmap=cmap, clim = clim, origin=origin, colorbar=colorbar)  
        
        
    def image_list(self, Vs = None, policies = None, episodes = None, cmap = 'auto', clim = None, origin='lower'):
        """
        Creates a sequence of images, one for each episode.
        """
        n_states = len(self.states())
        if Vs is not None:
            iterations = len(Vs)
        elif policies is not None:
            iterations = len(policies)
        else:
            iterations = len(episodes)
        
        V = None
        policy = None
        episode = None

        for i in range(iterations):
            if not Vs is None:
                V = Vs[i]
            if not policies is None:
                policy = policies[i]
            if not episodes is None:
                episode = episodes[i]    

            self.image(V, policy=policy, episode=episode, title=f'After Iteration {i}', cmap=cmap, clim = clim, origin=origin)  



def _image(m, labels=None, extra = None, title=None, cmap = 'auto', clim = None, origin='lower', colorbar=True):
    
    if isinstance(cmap, colors.Colormap):
        pass
    else: 
        if cmap == 'auto':      
            if (np.any(m < 0.0) and np.any(m > 0.0)) or (not clim is None and clim[0]<0 and clim[1]>0):
                cmap = "coolwarm"
            else:
                cmap = "Reds"

        cmap = plt.colormaps[cmap].copy()
        cmap.set_bad(color='black')
    
    row_labels = range(m.shape[0])
    col_labels = range(m.shape[1])
    
    fig, ax = plt.subplots()

    
    num_rows, num_cols = m.shape

    if not clim is None:
        im = ax.imshow(m, cmap=cmap, interpolation="nearest", origin=origin, vmin = clim[0], vmax=clim[1])
    else:
        im = ax.imshow(m, cmap=cmap, interpolation="nearest", origin=origin)
    
    ax.set_aspect("equal")
    
    # major tickmarks at the center of each cell for labels
    ax.set_xticks(np.arange(m.shape[1]))
    ax.set_yticks(np.arange(m.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # minor tickmarks at the edges of the cells for gridlines
    ax.set_xticks(np.arange(-.5, num_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, num_rows, 1), minor=True)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=.5)
    
    if not extra is None:
        for (j, i), label in np.ndenumerate(extra):
            ax.text(i, j, label, ha='center', va='center', color='grey', fontsize=15, fontweight='bold')
    
    if not labels is None:
        for (j, i), label in np.ndenumerate(labels):
            ax.text(i, j, label, ha='center', va='center', color='black', fontsize=10)
            

    if colorbar:
        plt.colorbar(im, ax=ax)
    
    plt.title(title)
    plt.show()


def parse_gridworld(layout_string):
    """Parse the layout string"""
    layout_string = layout_string.replace(
        '|', '')  # Remove optional pipe characters
    lines = layout_string.split('\n')
    lines = [l for l in lines if l != '']  # Remove empty lines

    # Get dimensions: assume rectangular (width, height)
    H = len(lines)
    W = len(lines[0])
    for l in lines:
        assert len(l) == W, "layout string is not rectangular; check dimensions"
    dims = (W, H)

    starts = list() # so we can sample from it
    
    # all others are hashed into sets for fast lookup
    goals = set()
    blocks = set()
    extra_labels = set()

    for row in range(H):
        for col in range(W):
            # Makes (0,0) the bottom-left cell in the gridworld
            coords = (col, H - 1 - row)
            char = lines[row][col]

            if char == 'S':  # Start (may be more than one)
                starts.append(coords)
            elif char == 'G':  # Goal (may be more than one)
                goals.add(coords)
            elif char == 'X':  # Block (agent cannot occupy these cells)
                blocks.add(coords)
            elif char == ' ':  # Empty (agent can occupy these cells)
                pass
            else:
                extra_labels.add((coords, char))
            #    raise ValueError(f"invalid character '{char}' at {coords}")

    return dims, starts, goals, blocks, extra_labels
