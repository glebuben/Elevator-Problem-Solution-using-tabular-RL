"""State-to-index encoding for the Q-table."""


def _build_cc_np_lookup():
    pairs = []
    for cc in range(32):
        pop = bin(cc).count('1')
        for np_ in range(pop, 6):
            pairs.append((cc, np_))
    return pairs, {p: i for i, p in enumerate(pairs)}


VALID_CC_NP_PAIRS, CC_NP_LOOKUP = _build_cc_np_lookup()
NUM_CC_NP = len(VALID_CC_NP_PAIRS)
TOTAL_STATES = 5 * 3 * 16 * 16 * NUM_CC_NP


def state_to_index(state):
    cf, cd, uc, dc, cc, np_ = state
    idx = cf * 3 + cd
    idx = idx * 16 + uc
    idx = idx * 16 + dc
    idx = idx * NUM_CC_NP + CC_NP_LOOKUP[(cc, np_)]
    return idx