from ..env.constants import (
    MOVE_UP, MOVE_DOWN, STOP, WAIT,
    DIR_UP, DIR_DOWN, DIR_IDLE,
)
from ..utils.helpers import (
    has_up_call, has_down_call, has_car_call,
)


class SCANPolicy:
    def __init__(self):
        self.sweep_dir = DIR_UP

    def select_action(self, state, valid_actions):
        cf, cd, uc, dc, cc, np_ = state

        if STOP in valid_actions:
            if has_car_call(cc, cf):
                return STOP
            if np_ < 5:
                if (self.sweep_dir == DIR_UP
                        and has_up_call(uc, cf)):
                    return (STOP
                            if cd in (DIR_UP, DIR_IDLE) else WAIT)
                if (self.sweep_dir == DIR_DOWN
                        and has_down_call(dc, cf)):
                    return (STOP
                            if cd in (DIR_DOWN, DIR_IDLE)
                            else WAIT)

        if self.sweep_dir == DIR_UP:
            if MOVE_UP in valid_actions:
                return MOVE_UP
            self.sweep_dir = DIR_DOWN
            if STOP in valid_actions:
                if has_car_call(cc, cf):
                    return STOP
                if np_ < 5:
                    if has_down_call(dc, cf):
                        return (STOP
                                if cd in (DIR_DOWN, DIR_IDLE)
                                else WAIT)
                    if has_up_call(uc, cf):
                        return (STOP
                                if cd in (DIR_UP, DIR_IDLE)
                                else WAIT)
            if MOVE_DOWN in valid_actions:
                return MOVE_DOWN
            return WAIT
        else:
            if MOVE_DOWN in valid_actions:
                return MOVE_DOWN
            self.sweep_dir = DIR_UP
            if STOP in valid_actions:
                if has_car_call(cc, cf):
                    return STOP
                if np_ < 5:
                    if has_up_call(uc, cf):
                        return (STOP
                                if cd in (DIR_UP, DIR_IDLE)
                                else WAIT)
                    if has_down_call(dc, cf):
                        return (STOP
                                if cd in (DIR_DOWN, DIR_IDLE)
                                else WAIT)
            if MOVE_UP in valid_actions:
                return MOVE_UP
            return WAIT