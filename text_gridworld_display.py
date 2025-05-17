# text_grid_world_display.py
# -----------------------
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


import math
import util
from functools import reduce

class TextGridworldDisplay:

    def __init__(self, grid_world):
        self.grid_world = grid_world

    def start(self):
        pass

    def pause(self):
        pass

    def display_values(self, agent, current_state = None, message = None):
        if message != None:
            print(message)
        values = util.Counter()
        policy = {}
        states = self.grid_world.get_states()
        for state in states:
            values[state] = agent.get_value(state)
            policy[state] = agent.get_policy(state)
        pretty_print_values(self.grid_world, values, policy, current_state)

    def display_null_values(self, agent, current_state = None, message = None):
        if message != None: print(message)
        pretty_print_null_values(self.grid_world, current_state)

    def display_q_values(self, agent, current_state = None, message = None):
        if message != None: print(message)
        q_values = util.Counter()
        states = self.grid_world.get_states()
        for state in states:
            for action in self.grid_world.get_possible_actions(state):
                q_values[(state, action)] = agent.get_q_value(state, action)
        pretty_print_q_values(self.grid_world, q_values, current_state)


def pretty_print_values(grid_world, values, policy=None, current_state = None):
    grid = grid_world.grid
    max_len = 11
    new_rows = []
    for y in range(grid.height):
        new_wow = []
        for x in range(grid.width):
            state = (x, y)
            value = values[state]
            action = None
            if policy != None and state in policy:
                action = policy[state]
            actions = grid_world.get_possible_actions(state)
            if action not in actions and 'exit' in actions:
                action = 'exit'
            val_string = None
            if action == 'exit':
                val_string = border('%.2f' % value)
            else:
                val_string = '\n\n%.2f\n\n' % value
                val_string += ' '*max_len
            if grid[x][y] == 'S':
                val_string = '\n\nS: %.2f\n\n'  % value
                val_string += ' '*max_len
            if grid[x][y] == '#':
                val_string = '\n#####\n#####\n#####\n'
                val_string += ' '*max_len
            pieces = [val_string]
            text = ("\n".join(pieces)).split('\n')
            if current_state == state:
                l = len(text[1])
                if l == 0:
                    text[1] = '*'
                else:
                    text[1] = "|" + ' ' * int((l-1)/2-1) + '*' + ' ' * int((l)/2-1) + "|"
            if action == 'east':
                text[2] = '  ' + text[2]  + ' >'
            elif action == 'west':
                text[2] = '< ' + text[2]  + '  '
            elif action == 'north':
                text[0] = ' ' * int(max_len/2) + '^' +' ' * int(max_len/2)
            elif action == 'south':
                text[4] = ' ' * int(max_len/2) + 'v' +' ' * int(max_len/2)
            new_cell = "\n".join(text)
            new_row.append(new_cell)
        new_rows.append(new_row)
    num_cols = grid.width
    for row_num, row in enumerate(new_rows):
        row.insert(0,"\n\n"+str(row_num))
    new_rows.reverse()
    col_labels = [str(col_num) for col_num in range(num_cols)]
    col_labels.insert(0,' ')
    final_rows = [col_labels] + new_rows
    print(indent(final_rows,separate_rows=True,delim='|', prefix='|',postfix='|', justify='center',has_header=True))


def pretty_print_null_values(grid_world, current_state = None):
    grid = grid_world.grid
    max_len = 11
    new_rows = []
    for y in range(grid.height):
        new_wow = []
        for x in range(grid.width):
            state = (x, y)

            # value = values[state]

            action = None
            # if policy != None and state in policy:
            #   action = policy[state]
            #
            actions = grid_world.get_possible_actions(state)

            if action not in actions and 'exit' in actions:
                action = 'exit'

            val_string = None
            # if action == 'exit':
            #   val_string = border('%.2f' % value)
            # else:
            #   val_string = '\n\n%.2f\n\n' % value
            #   val_string += ' '*max_len

            if grid[x][y] == 'S':
                val_string = '\n\nS\n\n'
                val_string += ' '*max_len
            elif grid[x][y] == '#':
                val_string = '\n#####\n#####\n#####\n'
                val_string += ' '*max_len
            elif type(grid[x][y]) == float or type(grid[x][y]) == int:
                val_string = border('%.2f' % float(grid[x][y]))
            else: val_string = border('  ')
            pieces = [val_string]

            text = ("\n".join(pieces)).split('\n')

            if current_state == state:
                l = len(text[1])
                if l == 0:
                    text[1] = '*'
                else:
                    text[1] = "|" + ' ' * int((l-1)/2-1) + '*' + ' ' * int((l)/2-1) + "|"

            if action == 'east':
                text[2] = '  ' + text[2]  + ' >'
            elif action == 'west':
                text[2] = '< ' + text[2]  + '  '
            elif action == 'north':
                text[0] = ' ' * int(max_len/2) + '^' +' ' * int(max_len/2)
            elif action == 'south':
                text[4] = ' ' * int(max_len/2) + 'v' +' ' * int(max_len/2)
            new_cell = "\n".join(text)
            new_row.append(new_cell)
        new_rows.append(new_row)
    num_cols = grid.width
    for row_num, row in enumerate(new_rows):
        row.insert(0,"\n\n"+str(row_num))
    new_rows.reverse()
    col_labels = [str(col_num) for col_num in range(num_cols)]
    col_labels.insert(0,' ')
    final_rows = [col_labels] + new_rows
    print(indent(final_rows,separate_rows=True,delim='|', prefix='|',postfix='|', justify='center',has_header=True))

def pretty_print_q_values(grid_world, q_values, current_state=None):
    grid = grid_world.grid
    max_len = 11
    new_rows = []
    for y in range(grid.height):
        new_row = []
        for x in range(grid.width):
            state = (x, y)
            actions = grid_world.get_possible_actions(state)
            if actions == None or len(actions) == 0:
                actions = [None]
            best_q = max([q_values[(state, action)] for action in actions])
            best_actions = [action for action in actions if q_values[(state, action)] == best_q]

            # display cell
            q_strings = dict([(action, "%.2f" % q_values[(state, action)]) for action in actions])
            north_string = ('north' in q_strings and q_strings['north']) or ' '
            south_string = ('south' in q_strings and q_strings['south']) or ' '
            east_string = ('east' in q_strings and q_strings['east']) or ' '
            west_string = ('west' in q_strings and q_strings['west']) or ' '
            exit_string = ('exit' in q_strings and q_strings['exit']) or ' '

            east_len = len(east_string)
            west_len = len(west_string)
            if east_len < west_len:
                east_string = ' '*(west_len-east_len)+east_string
            if west_len < east_len:
                west_string = west_string+' '*(east_len-west_len)

            if 'north' in best_actions:
                north_string = '/'+north_string+'\\'
            if 'south' in best_actions:
                south_string = '\\'+south_string+'/'
            if 'east' in best_actions:
                east_string = ''+east_string+'>'
            else:
                east_string = ''+east_string+' '
            if 'west' in best_actions:
                west_string = '<'+west_string+''
            else:
                west_string = ' '+west_string+''
            if 'exit' in best_actions:
                exit_string = '[ '+exit_string+' ]'


            ew_string = west_string + "     " + east_string
            if state == current_state:
                ew_string = west_string + "  *  " + east_string
            if state == grid_world.get_start_state():
                ew_string = west_string + "  S  " + east_string
            if state == current_state and state == grid_world.get_start_state():
                ew_string = west_string + " S:* " + east_string

            text = [north_string, "\n"+exit_string, ew_string, ' '*max_len+"\n", south_string]

            if grid[x][y] == '#':
                text = ['', '\n#####\n#####\n#####', '']

            new_cell = "\n".join(text)
            new_row.append(new_cell)
        new_rows.append(new_row)
    num_cols = grid.width
    for row_num, row in enumerate(new_rows):
        row.insert(0,"\n\n\n"+str(row_num))
    new_rows.reverse()
    col_labels = [str(col_num) for col_num in range(num_cols)]
    col_labels.insert(0,' ')
    final_rows = [col_labels] + new_rows

    print(indent(final_rows,separate_rows=True,delim='|',prefix='|',postfix='|', justify='center',has_header=True))

def border(text):
    length = len(text)
    pieces = ['-' * (length+2), '|'+' ' * (length+2)+'|', ' | '+text+' | ', '|'+' ' * (length+2)+'|','-' * (length+2)]
    return '\n'.join(pieces)

# INDENTING CODE

# Indenting code based on a post from George Sakkis
# (http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/267662)

import io,operator

def indent(rows, has_header=False, header_char='-', delim=' | ', justify='left',
           separate_rows=False, prefix='', postfix='', wrapfunc=lambda x:x):
    """Indents a table by column.
       - rows: A sequence of sequences of items, one sequence per row.
       - has_header: True if the first row consists of the columns' names.
       - header_char: Character to be used for the row separator line
         (if has_header==True or separateRows==True).
       - delim: The column delimiter.
       - justify: Determines how are data justified in their column.
         Valid values are 'left','right' and 'center'.
       - separate_rows: True if rows are to be separated by a line
         of 'header_char's.
       - prefix: A string prepended to each printed row.
       - postfix: A string appended to each printed row.
       - wrapfunc: A function f(text) for wrapping text; each element in
         the table is first wrapped by this function."""
    # closure for breaking logical rows to physical, using wrapfunc
    def row_wrapper(row):
        new_rows = [wrapfunc(item).split('\n') for item in row]
        return [[substr or '' for substr in item] for item in list(*new_rows)]
    # break each logical row into one or more physical ones
    logical_rows = [row_wrapper(row) for row in rows]
    # columns of physical rows
    columns = list(*reduce(operator.add,logical_rows))
    # get the maximum of each column by the string length of its items
    max_widths = [max([len(str(item)) for item in column]) for column in columns]
    row_separator = header_char * (len(prefix) + len(postfix) + sum(max_widths) + \
                                 len(delim)*(len(max_widths)-1))
    # select the appropriate justify method
    justify = {'center':str.center, 'right':str.rjust, 'left':str.ljust}[justify.lower()]
    output=io.StringIO()
    if separate_rows: print(row_separator, file=output)
    for physical_rows in logical_rows:
        for row in physical_rows:
            print(prefix \
                + delim.join([justify(str(item),width) for (item,width) in zip(row,max_widths)]) \
                + postfix, file=output)
        if separate_rows or has_header: print(row_separator, file=output); has_header=False
    return output.getvalue()


def wrap_always(text, width):
    """A simple word-wrap function that wraps text on exactly width characters.
       It doesn't split the text in words."""
    return '\n'.join([ text[width*i:width*(i+1)] \
                       for i in range(int(math.ceil(1.*len(text)/width))) ])


# TEST OF DISPLAY CODE

if __name__ == '__main__':
    import grid_world, util

    grid = grid_world.get_cliff_grid3()
    print(grid.get_states())

    policy = dict([(state,'east') for state in grid.get_states()])
    values = util.Counter(dict([(state,1000.23) for state in grid.get_states()]))
    pretty_print_values(grid, values, policy, current_state = (0,0))

    state_cross_actions = [[(state, action) for action in grid.get_possible_actions(state)] for state in grid.get_states()]
    q_states = reduce(lambda x,y: x+y, state_cross_actions, [])
    q_values = util.Counter(dict([((state, action), 10.5) for state, action in q_states]))
    q_values = util.Counter(dict([((state, action), 10.5) for state, action in reduce(lambda x,y: x+y, state_cross_actions, [])]))
    pretty_print_q_values(grid, q_values, current_state = (0,0))
