"""Elevator environment / simulator."""

import numpy as np
from collections import deque

from .passenger import Passenger
from .constants import (
    MOVE_UP, MOVE_DOWN, STOP, WAIT,
    DIR_DOWN, DIR_IDLE, DIR_UP,
)


class ElevatorEnv:
    def __init__(self, num_floors=5, capacity=5, lam=0.05,
                 traffic='uniform', seed=None):
        self.num_floors = num_floors
        self.capacity = capacity
        self.lam = lam
        self.traffic = traffic
        self.rng = np.random.RandomState(seed)
        self.up_queues = [deque() for _ in range(num_floors)]
        self.down_queues = [deque() for _ in range(num_floors)]
        self.car_passengers = []
        self.car_floor = 0
        self.car_direction = DIR_IDLE
        self.time_step = 0
        self.total_wait_time = 0
        self.total_system_time = 0
        self.total_served = 0
        self.total_boarded = 0
        self.up_button_since = [-1] * num_floors
        self.down_button_since = [-1] * num_floors
        self._setup_traffic()

    def _setup_traffic(self):
        nf = self.num_floors
        if self.traffic == 'uniform':
            self.floor_arrival_probs = np.ones(nf) / nf
            self.dest_probs = {}
            for f in range(nf):
                p = np.ones(nf)
                p[f] = 0.0
                p /= p.sum()
                self.dest_probs[f] = p
        elif self.traffic == 'up_peak':
            self.floor_arrival_probs = np.array(
                [0.6, 0.1, 0.1, 0.1, 0.1])
            self.dest_probs = {}
            for f in range(nf):
                if f == 0:
                    p = np.array([0.0, 0.15, 0.25, 0.30, 0.30])
                else:
                    p = np.ones(nf)
                    p[f] = 0.0
                    p /= p.sum()
                self.dest_probs[f] = p
        elif self.traffic == 'down_peak':
            self.floor_arrival_probs = np.array(
                [0.05, 0.15, 0.20, 0.30, 0.30])
            self.dest_probs = {}
            for f in range(nf):
                if f == 0:
                    p = np.ones(nf)
                    p[0] = 0.0
                    p /= p.sum()
                else:
                    p = np.zeros(nf)
                    p[0] = 0.7
                    for j in range(nf):
                        if j != f and j != 0:
                            p[j] = 0.3 / max(nf - 2, 1)
                    p[f] = 0.0
                    s = p.sum()
                    if s > 0:
                        p /= s
                self.dest_probs[f] = p
        elif self.traffic == 'mixed':
            self.floor_arrival_probs = np.array(
                [0.35, 0.15, 0.15, 0.15, 0.20])
            self.dest_probs = {}
            for f in range(nf):
                p = np.ones(nf)
                p[f] = 0.0
                if f != 0:
                    p[0] = 3.0
                p /= p.sum()
                self.dest_probs[f] = p
        else:
            raise ValueError(f"Unknown traffic: {self.traffic}")

    def reset(self):
        self.up_queues = [deque() for _ in range(self.num_floors)]
        self.down_queues = [deque() for _ in range(self.num_floors)]
        self.car_passengers = []
        self.car_floor = 0
        self.car_direction = DIR_IDLE
        self.time_step = 0
        self.total_wait_time = 0
        self.total_system_time = 0
        self.total_served = 0
        self.total_boarded = 0
        self.up_button_since = [-1] * self.num_floors
        self.down_button_since = [-1] * self.num_floors
        return self.get_state()

    def get_up_hall_calls(self):
        mask = 0
        for i in range(4):
            if len(self.up_queues[i]) > 0:
                mask |= (1 << i)
        return mask

    def get_down_hall_calls(self):
        mask = 0
        for i in range(4):
            if len(self.down_queues[i + 1]) > 0:
                mask |= (1 << i)
        return mask

    def get_car_calls(self):
        mask = 0
        for p in self.car_passengers:
            mask |= (1 << p.destination)
        return mask

    def get_num_passengers(self):
        return len(self.car_passengers)

    def get_state(self):
        return (self.car_floor, self.car_direction,
                self.get_up_hall_calls(), self.get_down_hall_calls(),
                self.get_car_calls(), self.get_num_passengers())

    def get_valid_actions(self):
        actions = [WAIT]
        if self.car_floor < self.num_floors - 1:
            actions.append(MOVE_UP)
        if self.car_floor > 0:
            actions.append(MOVE_DOWN)
        floor = self.car_floor
        has_exit = any(
            p.destination == floor for p in self.car_passengers)
        has_up = len(self.up_queues[floor]) > 0
        has_down = len(self.down_queues[floor]) > 0
        if has_exit or has_up or has_down:
            actions.append(STOP)
        return actions

    def _compute_observable_reward(self):
        reward = 0
        for f in range(self.num_floors):
            if self.up_button_since[f] >= 0:
                reward -= (self.time_step - self.up_button_since[f])
            if self.down_button_since[f] >= 0:
                reward -= (self.time_step - self.down_button_since[f])
        cc = self.get_car_calls()
        for f in range(5):
            if (cc >> f) & 1:
                reward -= 1
        return reward

    def _update_button_timestamps(self):
        for f in range(4):
            if len(self.up_queues[f]) > 0:
                if self.up_button_since[f] < 0:
                    self.up_button_since[f] = self.time_step
            else:
                self.up_button_since[f] = -1
        for f in range(1, 5):
            if len(self.down_queues[f]) > 0:
                if self.down_button_since[f] < 0:
                    self.down_button_since[f] = self.time_step
            else:
                self.down_button_since[f] = -1

    def step(self, action):
        valid = self.get_valid_actions()
        if action not in valid:
            action = WAIT

        if action == MOVE_UP:
            self.car_floor += 1
            self.car_direction = DIR_UP
        elif action == MOVE_DOWN:
            self.car_floor -= 1
            self.car_direction = DIR_DOWN
        elif action == WAIT:
            self.car_direction = DIR_IDLE
        elif action == STOP:
            remaining = []
            for p in self.car_passengers:
                if p.destination == self.car_floor:
                    self.total_system_time += (
                        self.time_step - p.arrival_time)
                    self.total_served += 1
                else:
                    remaining.append(p)
            self.car_passengers = remaining

            direction = self.car_direction
            if direction == DIR_UP:
                queues_to_serve = [self.up_queues[self.car_floor]]
            elif direction == DIR_DOWN:
                queues_to_serve = [
                    self.down_queues[self.car_floor]]
            else:
                queues_to_serve = [
                    self.up_queues[self.car_floor],
                    self.down_queues[self.car_floor]]
            for queue in queues_to_serve:
                while (queue
                       and len(self.car_passengers) < self.capacity):
                    pax = queue.popleft()
                    self.total_wait_time += (
                        self.time_step - pax.arrival_time)
                    self.total_boarded += 1
                    pax.board_time = self.time_step
                    self.car_passengers.append(pax)

        self._generate_arrivals()
        self._update_button_timestamps()
        reward = self._compute_observable_reward()
        self.time_step += 1
        return self.get_state(), reward

    def _generate_arrivals(self):
        for f in range(self.num_floors):
            rate = (self.lam * self.floor_arrival_probs[f]
                    * self.num_floors)
            n = self.rng.poisson(rate)
            for _ in range(n):
                dest = self.rng.choice(
                    self.num_floors, p=self.dest_probs[f])
                if dest == f:
                    continue
                p = Passenger(f, dest, self.time_step)
                if dest > f:
                    self.up_queues[f].append(p)
                else:
                    self.down_queues[f].append(p)

    def get_pending_metrics(self):
        pw_time = pw_n = ps_time = ps_n = 0
        for f in range(self.num_floors):
            for p in self.up_queues[f]:
                pw_time += self.time_step - p.arrival_time
                pw_n += 1
            for p in self.down_queues[f]:
                pw_time += self.time_step - p.arrival_time
                pw_n += 1
        for p in self.car_passengers:
            ps_time += self.time_step - p.arrival_time
            ps_n += 1
        return pw_time, pw_n, ps_time, ps_n