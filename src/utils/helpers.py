"""Call-detection helpers operating on bitmask state components."""


def has_up_call(uc, f):
    return f <= 3 and bool((uc >> f) & 1)


def has_down_call(dc, f):
    return f >= 1 and bool((dc >> (f - 1)) & 1)


def has_car_call(cc, f):
    return bool((cc >> f) & 1)


def has_any_call(uc, dc, cc, f):
    return (has_up_call(uc, f)
            or has_down_call(dc, f)
            or has_car_call(cc, f))


def calls_above(uc, dc, cc, cf):
    return any(
        has_any_call(uc, dc, cc, f) for f in range(cf + 1, 5))


def calls_below(uc, dc, cc, cf):
    return any(has_any_call(uc, dc, cc, f) for f in range(cf))


def calls_exist(uc, dc, cc):
    return uc != 0 or dc != 0 or cc != 0