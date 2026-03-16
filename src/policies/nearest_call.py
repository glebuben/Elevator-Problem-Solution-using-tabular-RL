from ..env.constants import (
    MOVE_UP, MOVE_DOWN, STOP, WAIT,
    DIR_UP, DIR_DOWN, DIR_IDLE,
)
from ..utils.helpers import (
    has_up_call, has_down_call, has_car_call, calls_exist,
)


class NearestCallPolicy:
    def __init__(self, rng):
        self.rng = rng

    def select_action(self, state, valid_actions):
        cf, cd, uc, dc, cc, np_ = state

        if STOP in valid_actions and has_car_call(cc, cf):
            return STOP
        if STOP in valid_actions and np_ < 5:
            hu = has_up_call(uc, cf)
            hd = has_down_call(dc, cf)
            if hu or hd:
                if cd == DIR_IDLE:
                    return STOP
                if cd == DIR_UP and hu:
                    return STOP
                if cd == DIR_DOWN and hd:
                    return STOP
                return WAIT

        if not calls_exist(uc, dc, cc):
            return WAIT

        target = None
        best_d = 999
        for f in range(5):
            if has_car_call(cc, f):
                d = abs(f - cf)
                if d < best_d:
                    best_d = d
                    target = f
        if target is None:
            for f in range(5):
                if (has_up_call(uc, f)
                        or has_down_call(dc, f)):
                    d = abs(f - cf)
                    if d < best_d:
                        best_d = d
                        target = f

        if target is None:
            return WAIT

        if target > cf:
            if (STOP in valid_actions and np_ < 5
                    and has_up_call(uc, cf)):
                return (STOP
                        if cd in (DIR_UP, DIR_IDLE) else WAIT)
            return (MOVE_UP
                    if MOVE_UP in valid_actions else WAIT)
        elif target < cf:
            if (STOP in valid_actions and np_ < 5
                    and has_down_call(dc, cf)):
                return (STOP
                        if cd in (DIR_DOWN, DIR_IDLE) else WAIT)
            return (MOVE_DOWN
                    if MOVE_DOWN in valid_actions else WAIT)
        else:
            if STOP in valid_actions:
                if cd == DIR_IDLE:
                    return STOP
                hu = has_up_call(uc, cf)
                hd = has_down_call(dc, cf)
                if cd == DIR_UP and hu:
                    return STOP
                if cd == DIR_DOWN and hd:
                    return STOP
                return WAIT
            return WAIT