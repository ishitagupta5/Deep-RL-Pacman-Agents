# grid_world.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import graphics_utils
import gridworld
import project3.value_iteration_agents as value_iteration_agents
import project3.q_learning_agents as q_learning_agents
import text_gridworld_display
import graphics_gridworld_display
    
import random
import sys
import mdp
import environment
import util
import optparse

class Gridworld(mdp.MarkovDecisionProcess):
    """
      Gridworld
    """
    def __init__(self, grid):
        # layout
        if type(grid) == type([]): grid = make_grid(grid)
        self.grid = grid

        # parameters
        self.living_reward = 0.0
        self.noise = 0.2
        # self.noise = 0

    def set_living_reward(self, reward):
        """
        The (negative) reward for exiting "normal" states.

        Note that in the R+N text, this reward is on entering
        a state and therefore is not clearly part of the state's
        future rewards.
        """
        self.living_reward = reward

    def set_noise(self, noise):
        """
        The probability of moving in an unintended direction.
        """
        self.noise = noise


    def get_possible_actions(self, state):
        """
        Returns list of valid actions for 'state'.

        Note that you can request moves into walls and
        that "exit" states transition to the terminal
        state under the special action "done".
        """
        if state == self.grid.terminal_state:
            return ()
        x,y = state
        if type(self.grid[x][y]) == int:
            return ('exit',)
        return ('north','west','south','east')

    def get_4_actions(self, state):
        actions_list = list(self.get_possible_actions(state))
        if len(actions_list) == 1:
            actions_list = [actions_list[0] for _ in range(4)]
        return actions_list

    def get_states(self):
        """
        Return list of all states.
        """
        # The true terminal state.
        states = [self.grid.terminal_state]
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if self.grid[x][y] != '#':
                    state = (x,y)
                    states.append(state)
        return states

    def get_reward(self, state, action, next_state):
        """
        Get reward for state, action, next_state transition.

        Note that the reward depends only on the state being
        departed (as in the R+N book examples, which more or
        less use this convention).
        """
        if state == self.grid.terminal_state:
            return 0.0
        x, y = state
        cell = self.grid[x][y]
        if type(cell) == int or type(cell) == float:
            return cell
        return self.living_reward

    def get_start_state(self):
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if self.grid[x][y] == 'S':
                    return (x, y)
        raise Exception('Grid has no start state')

    def is_terminal(self, state):
        """
        Only the TERMINAL_STATE state is *actually* a terminal state.
        The other "exit" states are technically non-terminals with
        a single action "exit" which leads to the true terminal state.
        This convention is to make the grids line up with the examples
        in the R+N textbook.
        """
        return state == self.grid.terminal_state


    def get_transition_states_and_probs(self, state, action):
        """
        Returns list of (next_state, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.
        """

        if action not in self.get_possible_actions(state):
            raise Exception("Illegal action!")

        if self.is_terminal(state):
            return []

        x, y = state

        if type(self.grid[x][y]) == int or type(self.grid[x][y]) == float:
            term_state = self.grid.terminal_state
            return [(term_state, 1.0)]

        successors = []

        north_state = (self.__is_allowed(y+1,x) and (x,y+1)) or state
        west_state = (self.__is_allowed(y,x-1) and (x-1,y)) or state
        south_state = (self.__is_allowed(y-1,x) and (x,y-1)) or state
        east_state = (self.__is_allowed(y,x+1) and (x+1,y)) or state

        if action == 'north' or action == 'south':
            if action == 'north':
                successors.append((north_state,1-self.noise))
            else:
                successors.append((south_state,1-self.noise))

            mass_left = self.noise
            successors.append((west_state,mass_left/2.0))
            successors.append((east_state,mass_left/2.0))

        if action == 'west' or action == 'east':
            if action == 'west':
                successors.append((west_state,1-self.noise))
            else:
                successors.append((east_state,1-self.noise))

            mass_left = self.noise
            successors.append((north_state,mass_left/2.0))
            successors.append((south_state,mass_left/2.0))

        successors = self.__aggregate(successors)

        return successors

    def __aggregate(self, states_and_probs):
        counter = util.Counter()
        for state, prob in states_and_probs:
            counter[state] += prob
        new_states_and_probs = []
        for state, prob in list(counter.items()):
            new_states_and_probs.append((state, prob))
        return new_states_and_probs

    def __is_allowed(self, y, x):
        if y < 0 or y >= self.grid.height: return False
        if x < 0 or x >= self.grid.width: return False
        return self.grid[x][y] != '#'

class GridworldEnvironment(environment.Environment):

    def __init__(self, grid_world):
        self.grid_world = grid_world
        self.reset()

    def get_current_state(self):
        return self.state

    def get_possible_actions(self, state):
        return self.grid_world.get_possible_actions(state)

    def do_action(self, action):
        state = self.get_current_state()
        (next_state, reward) = self.get_random_next_state(state, action)
        self.state = next_state
        return (next_state, reward)

    def get_random_next_state(self, state, action, rand_obj=None):
        rand = -1.0
        if rand_obj is None:
            rand = random.random()
        else:
            rand = rand_obj.random()
        sum = 0.0
        successors = self.grid_world.get_transition_states_and_probs(state, action)
        for next_state, prob in successors:
            sum += prob
            if sum > 1.0:
                raise Exception('Total transition probability more than one; sample failure.')
            if rand < sum:
                reward = self.grid_world.get_reward(state, action, next_state)
                return (next_state, reward)
        raise Exception('Total transition probability less than one; sample failure.')

    def reset(self):
        self.state = self.grid_world.get_start_state()

class Grid:
    """
    A 2-dimensional array of immutables backed by a list of lists.  Data is accessed
    via grid[x][y] where (x,y) are cartesian coordinates with x horizontal,
    y vertical and the origin (0,0) in the bottom left corner.

    The __str__ method constructs an output that is oriented appropriately.
    """
    def __init__(self, width, height, initial_value=' '):
        self.width = width
        self.height = height
        self.data = [[initial_value for y in range(height)] for x in range(width)]
        self.terminal_state = 'TERMINAL_STATE'

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __eq__(self, other):
        if other == None: return False
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def copy(self):
        g = Grid(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def deep_copy(self):
        return self.copy()

    def shallow_copy(self):
        g = Grid(self.width, self.height)
        g.data = self.data
        return g

    def _get_legacy_text(self):
        t = [[self.data[x][y] for x in range(self.width)] for y in range(self.height)]
        t.reverse()
        return t

    def __str__(self):
        return str(self._get_legacy_text())

def make_grid(grid_string):
    width, height = len(grid_string[0]), len(grid_string)
    grid = Grid(width, height)
    for ybar, line in enumerate(grid_string):
        y = height - ybar - 1
        for x, el in enumerate(line):
            grid[x][y] = el
    return grid

def get_cliff_grid():
    grid = [[' ',' ',' ',' ',' '],
            ['S',' ',' ',' ',10],
            [-100,-100, -100, -100, -100]]
    return Gridworld(make_grid(grid))

def get_cliff_grid2():
    grid = [[' ',' ',' ',' ',' '],
            [8,'S',' ',' ',10],
            [-100,-100, -100, -100, -100]]
    return Gridworld(grid)

def get_discount_grid():
    grid = [[' ',' ',' ',' ',' '],
            [' ','#',' ',' ',' '],
            [' ','#', 1,'#', 10],
            ['S',' ',' ',' ',' '],
            [-10,-10, -10, -10, -10]]
    return Gridworld(grid)

def get_bridge_grid():
    grid = [[ '#',-100, -100, -100, -100, -100, '#'],
            [   1, 'S',  ' ',  ' ',  ' ',  ' ',  10],
            [ '#',-100, -100, -100, -100, -100, '#']]
    return Gridworld(grid)

def get_book_grid():
    grid = [[' ',' ',' ',+1],
            [' ','#',' ',-1],
            ['S',' ',' ',' ']]
    return Gridworld(grid)

def get_maze_grid():
    grid = [[' ',' ',' ',+1],
            ['#','#',' ','#'],
            [' ','#',' ',' '],
            [' ','#','#',' '],
            ['S',' ',' ',' ']]
    return Gridworld(grid)



def get_user_action(state, action_function):
    """
    Get an action from the user (rather than the agent).

    Used for debugging and lecture demos.
    """
    
    action = None
    while True:
        keys = graphics_utils.wait_for_keys()
        if 'Up' in keys: action = 'north'
        if 'Down' in keys: action = 'south'
        if 'Left' in keys: action = 'west'
        if 'Right' in keys: action = 'east'
        if 'q' in keys: sys.exit(0)
        if action == None: continue
        break
    actions = action_function(state)
    if action not in actions:
        action = actions[0]
    return action

def print_string(x): print(x)

def run_episode(agent, environment, discount, decision, display, message, pause, episode):
    returns = 0
    total_discount = 1.0
    environment.reset()
    if 'start_episode' in dir(agent): agent.start_episode()
    message("BEGINNING EPISODE: "+str(episode)+"\n")
    while True:

        # DISPLAY CURRENT STATE
        state = environment.get_current_state()
        display(state)
        pause()

        # END IF IN A TERMINAL STATE
        actions = environment.get_possible_actions(state)
        if len(actions) == 0:
            message("EPISODE "+str(episode)+" COMPLETE: RETURN WAS "+str(returns)+"\n")
            return returns

        # GET ACTION (USUALLY FROM AGENT)
        action = decision(state)
        if action == None:
            raise Exception('Error: Agent returned None action')

        # EXECUTE ACTION
        next_state, reward = environment.do_action(action)
        message("Started in state: "+str(state)+
                "\nTook action: "+str(action)+
                "\nEnded in state: "+str(next_state)+
                "\nGot reward: "+str(reward)+"\n")
        # UPDATE LEARNER
        if 'observe_transition' in dir(agent):
            agent.observe_transition(state, action, next_state, reward)

        returns += reward * total_discount
        total_discount *= discount

    if 'stop_episode' in dir(agent):
        agent.stop_episode()

def parse_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('-d', '--discount',action='store',
                         type='float',dest='discount',default=0.9,
                         help='Discount on future (default %default)')
    opt_parser.add_option('-r', '--living_reward',action='store',
                         type='float',dest='living_reward',default=0.0,
                         metavar="R", help='Reward for living for a time step (default %default)')
    opt_parser.add_option('-n', '--noise',action='store',
                         type='float',dest='noise',default=0.2,
                         metavar="P", help='How often action results in ' +
                         'unintended direction (default %default)' )
    opt_parser.add_option('-e', '--epsilon',action='store',
                         type='float',dest='epsilon',default=0.3,
                         metavar="E", help='Chance of taking a random action in q-learning (default %default)')
    opt_parser.add_option('-l', '--learning_rate',action='store',
                         type='float',dest='learning_rate',default=0.5,
                         metavar="P", help='TD learning rate (default %default)' )
    opt_parser.add_option('-i', '--iterations',action='store',
                         type='int',dest='iters',default=10,
                         metavar="K", help='Number of rounds of value iteration (default %default)')
    opt_parser.add_option('-k', '--episodes',action='store',
                         type='int',dest='episodes',default=1,
                         metavar="K", help='Number of epsiodes of the MDP to run (default %default)')
    opt_parser.add_option('-g', '--grid',action='store',
                         metavar="G", type='string',dest='grid',default="book_grid",
                         help='Grid to use (case sensitive; options are book_grid, bridge_grid, cliff_grid, maze_grid, default %default)' )
    opt_parser.add_option('-w', '--window_size', metavar="X", type='int',dest='grid_size',default=150,
                         help='Request a window width of X pixels *per grid cell* (default %default)')
    opt_parser.add_option('-a', '--agent',action='store', metavar="A",
                         type='string',dest='agent',default="random",
                         help='Agent type (options are \'random\', \'value\', \'q\', and \'learn\', default %default)')
    opt_parser.add_option('-t', '--text',action='store_true',
                         dest='text_display',default=False,
                         help='Use text-only ASCII display')
    opt_parser.add_option('-p', '--pause',action='store_true',
                         dest='pause',default=False,
                         help='Pause GUI after each time step when running the MDP')
    opt_parser.add_option('-q', '--quiet',action='store_true',
                         dest='quiet',default=False,
                         help='Skip display of any learning episodes')
    opt_parser.add_option('-s', '--speed',action='store', metavar="S", type=float,
                         dest='speed',default=1.0,
                         help='Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %default)')
    opt_parser.add_option('-m', '--manual',action='store_true',
                         dest='manual',default=False,
                         help='Manually control agent')
    opt_parser.add_option('-v', '--value_steps',action='store_true' ,default=False,
                         help='Display each step of value iteration')

    opts, args = opt_parser.parse_args()

    if opts.manual and (opts.agent != 'q' and opts.agent != 'learn'):
        print('## Disabling Agents in Manual Mode (-m) ##')
        opts.agent = None

    # MANAGE CONFLICTS
    if opts.text_display or opts.quiet:
    # if opts.quiet:
        opts.pause = False
        # opts.manual = False

    if opts.manual:
        opts.pause = True

    return opts


if __name__ == '__main__':

    opts = parse_options()

    ###########################
    # GET THE grid_world
    ###########################

    
    mdp_function = getattr(gridworld, "get_"+opts.grid)
    mdp = mdp_function()
    mdp.set_living_reward(opts.living_reward)
    mdp.set_noise(opts.noise)
    env = gridworld.GridworldEnvironment(mdp)


    ###########################
    # GET THE DISPLAY ADAPTER
    ###########################


    display = text_gridworld_display.TextGridworldDisplay(mdp)
    if not opts.text_display:
        display = graphics_gridworld_display.GraphicsGridworldDisplay(mdp, opts.grid_size, opts.speed)
    try:
        display.start()
    except KeyboardInterrupt:
        sys.exit(0)

    ###########################
    # GET THE AGENT
    ###########################

    
    a = None
    if opts.agent == 'value':
        a = value_iteration_agents.ValueIterationAgent(mdp, opts.discount, opts.iters)
    elif opts.agent == 'learn':
        print("HERE")
        grid_world_env = GridworldEnvironment(mdp)
        action_fn = lambda state: mdp.get_possible_actions(state)
        q_learn_opts = {'gamma': opts.discount,
                        'alpha': opts.learning_rate,
                        'epsilon': opts.epsilon,
                        'action_fn': action_fn}
        a = q_learning_agents.LearnedQAgent(grid_world_env.grid_world, **q_learn_opts)
    elif opts.agent == 'q':
        #env.get_possible_actions, opts.discount, opts.learningRate, opts.epsilon
        #simulation_fn = lambda agent, state: simulation.grid_worldSimulation(agent,state,mdp)
        grid_world_env = GridworldEnvironment(mdp)
        action_fn = lambda state: mdp.get_possible_actions(state)
        q_learn_opts = {'gamma': opts.discount,
                        'alpha': opts.learning_rate,
                        'epsilon': opts.epsilon,
                        'action_fn': action_fn}
        a = q_learning_agents.QLearningAgent(**q_learn_opts)
    elif opts.agent == 'random':
        # # No reason to use the random agent without episodes
        if opts.episodes == 0:
            opts.episodes = 10
        class RandomAgent:
            def get_action(self, state):
                return random.choice(mdp.get_possible_actions(state))
            def get_value(self, state):
                return 0.0
            def get_q_value(self, state, action):
                return 0.0
            def get_policy(self, state):
                "NOTE: 'random' is a special policy value; don't use it in your code."
                return 'random'
            def update(self, state, action, next_state, reward):
                pass
        a = RandomAgent()
    elif opts.agent == 'asynchvalue':
        a = value_iteration_agents.AsynchronousValueIterationAgent(mdp, opts.discount, opts.iters)
    elif opts.agent == 'priosweepvalue':
        a = value_iteration_agents.PrioritizedSweepingValueIterationAgent(mdp, opts.discount, opts.iters)
    else:
        if not opts.manual: raise Exception('Unknown agent type: '+opts.agent)


    ###########################
    # RUN EPISODES
    ###########################
    # DISPLAY Q/V VALUES BEFORE SIMULATION OF EPISODES
    try:
        if not opts.manual and opts.agent in ('value', 'asynchvalue', 'priosweepvalue', 'learn'):
            if opts.value_steps:
                for i in range(opts.iters):
                    temp_agent = value_iteration_agents.ValueIterationAgent(mdp, opts.discount, i)
                    display.display_values(temp_agent, message = "VALUES AFTER "+str(i)+" ITERATIONS")
                    display.pause()

            display.display_values(a, message = "VALUES AFTER "+str(opts.iters)+" ITERATIONS")
            display.pause()
            display.display_q_values(a, message = "Q-VALUES AFTER "+str(opts.iters)+" ITERATIONS")
            display.pause()
    except KeyboardInterrupt:
        sys.exit(0)



    # FIGURE OUT WHAT TO DISPLAY EACH TIME STEP (IF ANYTHING)
    display_callback = lambda x: None
    if not opts.quiet:
        if opts.manual and opts.agent == None:
            display_callback = lambda state: display.display_null_values(state)
        else:
            if opts.agent in ('random', 'value', 'asynchvalue', 'priosweepvalue'):
                display_callback = lambda state: display.display_values(a, state, "CURRENT VALUES")
            if opts.agent == 'q': display_callback = lambda state: display.display_q_values(a, state, "CURRENT Q-VALUES")

    message_callback = lambda x: print_string(x)
    if opts.quiet:
        message_callback = lambda x: None

    # FIGURE OUT WHETHER TO WAIT FOR A KEY PRESS AFTER EACH TIME STEP
    pause_callback = lambda : None
    if opts.pause:
        pause_callback = lambda : display.pause()

    # FIGURE OUT WHETHER THE USER WANTS MANUAL CONTROL (FOR DEBUGGING AND DEMOS)
    if opts.manual:
        decision_callback = lambda state : get_user_action(state, mdp.get_possible_actions)
    else:
        decision_callback = a.get_action

    # RUN EPISODES
    if opts.episodes > 0:
        print()
        print("RUNNING", opts.episodes, "EPISODES")
        print()
    returns = 0
    for episode in range(1, opts.episodes+1):
        returns += run_episode(a, env, opts.discount, decision_callback, display_callback, message_callback, pause_callback, episode)
    if opts.episodes > 0:
        print()
        print("AVERAGE RETURNS FROM START STATE: "+str((returns+0.0) / opts.episodes))
        print()
        print()

    # DISPLAY POST-LEARNING VALUES / Q-VALUES
    if opts.agent == 'q' and not opts.manual:
        try:
            display.display_q_values(a, message = "Q-VALUES AFTER "+str(opts.episodes)+" EPISODES")
            display.pause()
            display.display_values(a, message = "VALUES AFTER "+str(opts.episodes)+" EPISODES")
            display.pause()
        except KeyboardInterrupt:
            sys.exit(0)
