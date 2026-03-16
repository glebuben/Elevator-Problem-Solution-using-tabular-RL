from ..env.constants import (
    MOVE_UP, MOVE_DOWN, STOP, WAIT,
    DIR_UP, DIR_DOWN, DIR_IDLE,
)
from ..utils.helpers import (
    has_up_call, has_down_call, has_car_call,
    calls_above, calls_below, calls_exist,
)


class LOOKPolicy:
    def __init__(self):
        self.direction = DIR_UP

    def _try_stop(self, state, valid_actions):
        cf, cd, uc, dc, cc, np_ = state
        if STOP not in valid_actions:
            return None
        if has_car_call(cc, cf):
            return STOP
        if np_ >= 5:
            return None
        hu = has_up_call(uc, cf)
        hd = has_down_call(dc, cf)
        if not hu and not hd:
            return None
        if cd == DIR_IDLE:
            return STOP
        if cd == DIR_UP and hu:
            return STOP
        if cd == DIR_DOWN and hd:
            return STOP
        return WAIT

    def select_action(self, state, valid_actions):
        cf, cd, uc, dc, cc, np_ = state
        if not calls_exist(uc, dc, cc):
            return WAIT

        sa = self._try_stop(state, valid_actions)
        if sa is not None:
            return sa

        if self.direction == DIR_UP:
            if (calls_above(uc, dc, cc, cf)
                    and MOVE_UP in valid_actions):
                return MOVE_UP
            self.direction = DIR_DOWN
            sa = self._try_stop(state, valid_actions)
            if sa is not None:
                return sa
            if (calls_below(uc, dc, cc, cf)
                    and MOVE_DOWN in valid_actions):
                return MOVE_DOWN
            return WAIT
        else:
            if (calls_below(uc, dc, cc, cf)
                    and MOVE_DOWN in valid_actions):
                return MOVE_DOWN
            self.direction = DIR_UP
            sa = self._try_stop(state, valid_actions)
            if sa is not None:
                return sa
            if (calls_above(uc, dc, cc, cf)
                    and MOVE_UP in valid_actions):
                return MOVE_UP
            return WAIT