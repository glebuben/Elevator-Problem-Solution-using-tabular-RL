"""Passenger data class."""


class Passenger:
    __slots__ = ['origin', 'destination', 'arrival_time', 'board_time']

    def __init__(self, origin, destination, arrival_time):
        self.origin = origin
        self.destination = destination
        self.arrival_time = arrival_time
        self.board_time = None