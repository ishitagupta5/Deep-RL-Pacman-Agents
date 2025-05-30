# crawler.py
# ----------
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


#!/usr/bin/python
from graphics_crawler_display import *
import math
from math import pi as PI
import time
import environment
import random

class CrawlingRobotEnvironment(environment.Environment):

    def __init__(self, crawling_robot):

        self.crawling_robot = crawling_robot

        # The state is of the form (arm_angle, hand_angle)
        # where the angles are bucket numbers, not actual
        # degree measurements
        self.state = None

        self.n_arm_states = 9
        self.n_hand_states = 13

        # create a list of arm buckets and hand buckets to
        # discretize the state space
        (min_arm_angle,max_arm_angle) = self.crawling_robot.get_min_and_max_arm_angles()
        (min_hand_angle,max_hand_angle) = self.crawling_robot.get_min_and_max_hand_angles()
        arm_increment = (max_arm_angle - min_arm_angle) / (self.n_arm_states-1)
        hand_increment = (max_hand_angle - min_hand_angle) / (self.n_hand_states-1)
        self.arm_buckets = [min_arm_angle+(arm_increment*i) \
           for i in range(self.n_arm_states)]
        self.hand_buckets = [min_hand_angle+(hand_increment*i) \
         for i in range(self.n_hand_states)]

        # Reset
        self.reset()

    def get_current_state(self):
        """
          Return the current state
          of the crawling robot
        """
        return self.state

    def get_possible_actions(self, state):
        """
          Returns possible actions
          for the states in the
          current state
        """

        actions = list()

        (curr_arm_bucket,curr_hand_bucket) = state
        if curr_arm_bucket > 0: actions.append('arm-down')
        if curr_arm_bucket < self.n_arm_states-1: actions.append('arm-up')
        if curr_hand_bucket > 0: actions.append('hand-down')
        if curr_hand_bucket < self.n_hand_states-1: actions.append('hand-up')

        return actions

    def do_action(self, action):
        """
          Perform the action and update
          the current state of the Environment
          and return the reward for the
          current state, the next state
          and the taken action.

          Returns:
            next_state, reward
        """
        (next_state,reward) =  (None, None)

        old_x,old_y = self.crawling_robot.get_robot_position()

        (arm_bucket,hand_bucket) = self.state
        (arm_angle,hand_angle) = self.crawling_robot.get_angles()
        if action == 'arm-up':
            new_arm_angle = self.arm_buckets[arm_bucket+1]
            self.crawling_robot.move_arm(new_arm_angle)
            next_state = (arm_bucket+1,hand_bucket)
        if action == 'arm-down':
            new_arm_angle = self.arm_buckets[arm_bucket-1]
            self.crawling_robot.move_arm(new_arm_angle)
            next_state = (arm_bucket-1,hand_bucket)
        if action == 'hand-up':
            new_hand_angle = self.hand_buckets[hand_bucket+1]
            self.crawling_robot.move_hand(new_hand_angle)
            next_state = (arm_bucket,hand_bucket+1)
        if action == 'hand-down':
            new_hand_angle = self.hand_buckets[hand_bucket-1]
            self.crawling_robot.move_hand(new_hand_angle)
            next_state = (arm_bucket,hand_bucket-1)

        (new_x,new_y) = self.crawling_robot.get_robot_position()

        # a simple reward function
        reward = new_x - old_x

        self.state = next_state
        return (next_state, reward)


    def reset(self):
        """
         Resets the Environment to the initial state
        """
        ## Initialize the state to be the middle
        ## value for each parameter e.g. if there are 13 and 19
        ## buckets for the arm and hand parameters, then the intial
        ## state should be (6,9)
        ##
        ## Also call self.crawling_robot.set_angles()
        ## to the initial arm and hand angle

        arm_state = self.n_arm_states//2
        hand_state = self.n_hand_states//2
        self.state = (arm_state,hand_state)
        self.crawling_robot.set_angles(self.arm_buckets[arm_state],self.hand_buckets[hand_state])
        self.crawling_robot.positions = [20,self.crawling_robot.get_robot_position()[0]]


class CrawlingRobot:

    def set_angles(self, arm_angle, hand_angle):
        """
            set the robot's arm and hand angles
            to the passed in values
        """
        self.arm_angle = arm_angle
        self.hand_angle = hand_angle

    def get_angles(self):
        """
            returns the pair of (arm_angle, hand_angle)
        """
        return (self.arm_angle, self.hand_angle)

    def get_robot_position(self):
        """
            returns the (x,y) coordinates
            of the lower-left point of the
            robot
        """
        return self.robot_pos

    def move_arm(self, new_arm_angle):
        """
            move the robot arm to 'new_arm_angle'
        """
        old_arm_angle = self.arm_angle
        if new_arm_angle > self.max_arm_angle:
            raise Exception('Crawling Robot: Arm Raised too high. Careful!')
        if new_arm_angle < self.min_arm_angle:
            raise Exception('Crawling Robot: Arm Raised too low. Careful!')
        disp = self.displacement(self.arm_angle, self.hand_angle,
                                  new_arm_angle, self.hand_angle)
        cur_x_pos = self.robot_pos[0]
        self.robot_pos = (cur_x_pos+disp, self.robot_pos[1])
        self.arm_angle = new_arm_angle

        # Position and Velocity Sign Post
        self.positions.append(self.get_robot_position()[0])
#        self.angle_sums.append(abs(math.degrees(old_arm_angle)-math.degrees(new_arm_angle)))
        if len(self.positions) > 100:
            self.positions.pop(0)
 #           self.angle_sums.pop(0)

    def move_hand(self, new_hand_angle):
        """
            move the robot hand to 'new_arm_angle'
        """
        old_hand_angle = self.hand_angle

        if new_hand_angle > self.max_hand_angle:
            raise Exception('Crawling Robot: Hand Raised too high. Careful!')
        if new_hand_angle < self.min_hand_angle:
            raise Exception('Crawling Robot: Hand Raised too low. Careful!')
        disp = self.displacement(self.arm_angle, self.hand_angle, self.arm_angle, new_hand_angle)
        cur_x_pos = self.robot_pos[0]
        self.robot_pos = (cur_x_pos+disp, self.robot_pos[1])
        self.hand_angle = new_hand_angle

        # Position and Velocity Sign Post
        self.positions.append(self.get_robot_position()[0])
 #       self.angle_sums.append(abs(math.degrees(old_hand_angle)-math.degrees(new_hand_angle)))
        if len(self.positions) > 100:
            self.positions.pop(0)
 #           self.angle_sums.pop(0)

    def get_min_and_max_arm_angles(self):
        """
            get the lower- and upper- bound
            for the arm angles returns (min,max) pair
        """
        return (self.min_arm_angle, self.max_arm_angle)

    def get_min_and_max_hand_angles(self):
        """
            get the lower- and upper- bound
            for the hand angles returns (min,max) pair
        """
        return (self.min_hand_angle, self.max_hand_angle)

    def get_rotation_angle(self):
        """
            get the current angle the
            robot body is rotated off the ground
        """
        arm_cos, arm_sin = self.__get_cos_and_sin(self.arm_angle)
        hand_cos, hand_sin = self.__get_cos_and_sin(self.hand_angle)
        x = self.arm_length * arm_cos + self.hand_length * hand_cos + self.robot_width
        y = self.arm_length * arm_sin + self.hand_length * hand_sin + self.robot_height
        if y < 0:
            return math.atan(-y/x)
        return 0.0


    ## You shouldn't need methods below here


    def __get_cos_and_sin(self, angle):
        return math.cos(angle), math.sin(angle)

    def displacement(self, old_arm_degree, old_hand_degree, arm_degree, hand_degree):
        old_arm_cos, old_arm_sin = self.__get_cos_and_sin(old_arm_degree)
        arm_cos, arm_sin = self.__get_cos_and_sin(arm_degree)
        old_hand_cos, old_hand_sin = self.__get_cos_and_sin(old_hand_degree)
        hand_cos, hand_sin = self.__get_cos_and_sin(hand_degree)

        x_old = self.arm_length * old_arm_cos + self.hand_length * old_hand_cos + self.robot_width
        y_old = self.arm_length * old_arm_sin + self.hand_length * old_hand_sin + self.robot_height

        x = self.arm_length * arm_cos + self.hand_length * hand_cos + self.robot_width
        y = self.arm_length * arm_sin + self.hand_length * hand_sin + self.robot_height

        if y < 0:
            if y_old <= 0:
                return math.sqrt(x_old*x_old + y_old*y_old) - math.sqrt(x*x + y*y)
            return (x_old - y_old*(x-x_old) / (y - y_old)) - math.sqrt(x*x + y*y)
        else:
            if y_old  >= 0:
                return 0.0
            return -(x - y * (x_old-x)/(y_old-y)) + math.sqrt(x_old*x_old + y_old*y_old)

        raise Exception('Never Should See This!')

    def draw(self, step_count, step_delay):
        x1, y1 = self.get_robot_position()
        x1 = x1 % self.tot_width

        ## Check Lower Still on the ground
        if y1 != self.ground_y:
            raise Exception('Flying Robot!!')

        rotation_angle = self.get_rotation_angle()
        cos_rot, sin_rot = self.__get_cos_and_sin(rotation_angle)

        x2 = x1 + self.robot_width * cos_rot
        y2 = y1 - self.robot_width * sin_rot

        x3 = x1 - self.robot_height * sin_rot
        y3 = y1 - self.robot_height * cos_rot

        x4 = x3 + cos_rot*self.robot_width
        y4 = y3 - sin_rot*self.robot_width

        self.canvas.coords(self.robot_body,x1,y1,x2,y2,x4,y4,x3,y3)

        arm_cos, arm_sin = self.__get_cos_and_sin(rotation_angle+self.arm_angle)
        x_arm = x4 + self.arm_length * arm_cos
        y_arm = y4 - self.arm_length * arm_sin

        self.canvas.coords(self.robot_arm,x4,y4,x_arm,y_arm)

        hand_cos, hand_sin = self.__get_cos_and_sin(self.hand_angle+rotation_angle)
        x_hand = x_arm + self.hand_length * hand_cos
        y_hand = y_arm - self.hand_length * hand_sin

        self.canvas.coords(self.robot_hand,x_arm,y_arm,x_hand,y_hand)


        # Position and Velocity Sign Post
#        time = len(self.positions) + 0.5 * sum(self.angle_sums)
#        velocity = (self.positions[-1]-self.positions[0]) / time
#        if len(self.positions) == 1: return
        steps = (step_count - self.last_step)
        if steps==0:return
 #       pos = self.positions[-1]
#        velocity = (pos - self.last_pos) / steps
  #      g = .9 ** (10 * step_delay)
#        g = .99 ** steps
#        self.vel_avg = g * self.vel_avg + (1 - g) * velocity
 #       g = .999 ** steps
 #       self.vel_avg2 = g * self.vel_avg2 + (1 - g) * velocity
        pos = self.positions[-1]
        velocity = pos - self.positions[-2]
        vel2 = (pos - self.positions[0]) / len(self.positions)
        self.vel_avg = .9 * self.vel_avg + .1 * vel2
        vel_msg = '100-step Avg Velocity: %.2f' % self.vel_avg
#        vel_msg2 = '1000-step Avg Velocity: %.2f' % self.vel_avg2
        velocity_msg = 'Velocity: %.2f' % velocity
        position_msg = 'Position: %2.f' % pos
        stepMsg = 'Step: %d' % step_count
        if 'vel_msg' in dir(self):
            self.canvas.delete(self.vel_msg)
            self.canvas.delete(self.pos_msg)
            self.canvas.delete(self.step_msg)
            self.canvas.delete(self.velavg_msg)
 #           self.canvas.delete(self.velavg2_msg)
 #       self.velavg2_msg = self.canvas.create_text(850,190,text=vel_msg2)
        self.velavg_msg = self.canvas.create_text(650,190,text=vel_msg)
        self.vel_msg = self.canvas.create_text(450,190,text=velocity_msg)
        self.pos_msg = self.canvas.create_text(250,190,text=position_msg)
        self.step_msg = self.canvas.create_text(50,190,text=stepMsg)
#        self.last_pos = pos
        self.last_step = step_count
#        self.lastVel = velocity

    def __init__(self, canvas):

        ## Canvas ##
        self.canvas = canvas
        self.vel_avg = 0
#        self.vel_avg2 = 0
#        self.last_pos = 0
        self.last_step = 0
#        self.lastVel = 0

        ## Arm and Hand Degrees ##
        self.arm_angle = self.old_arm_degree = 0.0
        self.hand_angle = self.old_hand_degree = -PI/6

        self.max_arm_angle = PI/6
        self.min_arm_angle = -PI/6

        self.max_hand_angle = 0
        self.min_hand_angle = -(5.0/6.0) * PI

        ## Draw Ground ##
        self.tot_width = canvas.winfo_reqwidth()
        self.tot_height = canvas.winfo_reqheight()
        self.ground_height = 40
        self.ground_y = self.tot_height - self.ground_height

        self.ground = canvas.create_rectangle(0,
            self.ground_y,self.tot_width,self.tot_height, fill='blue')

        ## Robot Body ##
        self.robot_width = 80
        self.robot_height = 40
        self.robot_pos = (20, self.ground_y)
        self.robot_body = canvas.create_polygon(0,0,0,0,0,0,0,0, fill='green')

        ## Robot Arm ##
        self.arm_length = 60
        self.robot_arm = canvas.create_line(0,0,0,0,fill='orange',width=5)

        ## Robot Hand ##
        self.hand_length = 40
        self.robot_hand = canvas.create_line(0,0,0,0,fill='red',width=3)

        self.positions = [0,0]
  #      self.angle_sums = [0,0]



if __name__ == '__main__':
    run()
