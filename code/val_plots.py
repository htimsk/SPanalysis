## Code modified from  https://github.com/Valdorff/SPanalysis

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

DAYS_IN_AWARD_PERIOD = 28
AWARD_PERIODS_PER_YEAR = 365.25 / DAYS_IN_AWARD_PERIOD
SLOTS_PER_AWARD_PERIOD = DAYS_IN_AWARD_PERIOD * 24 * 60 * 60 / 12
NET_VALIDATORS = 425000  #number of validators from https://beaconcha.in/
FLASHBOTS_DATA = np.genfromtxt('blockReward.csv', delimiter=',')


def calc_ppv_ls(n_validators, n_years, n_trials, n_smoothie=None, per_validator=False):
    # treat solitarius as a case where we are all the validators; then the rest can be
    # handled the same for both solitarius and smoothifiers
    if n_smoothie is None:
        n_smoothie = n_validators

    # get expected number of proposals
    n_periods = n_years * AWARD_PERIODS_PER_YEAR
    net_slots = int(round(SLOTS_PER_AWARD_PERIOD * n_smoothie * n_periods))

    # scipy.stats uses int32s in it, so need to give it some distance from the max i32
    max_num = 2**30 - 1
    if net_slots > max_num:
        slots = max_num
        proposal_scalar = int(round(net_slots / slots))
    else:
        slots = net_slots
        proposal_scalar = 1

    proposals_ls = scipy.stats.binom.rvs(slots, (1 / NET_VALIDATORS), loc=0, size=n_trials)
    proposals_ls = [proposals * proposal_scalar for proposals in proposals_ls]

    total_rewards_ls = []
    for proposals in proposals_ls:
        block_reward_arr = np.random.choice(FLASHBOTS_DATA, proposals, replace=True)
        total_rewards_ls.append(np.sum(block_reward_arr) / 1e18)  # convert wei to ETH

    if per_validator:
        reward_scalar = 1 / n_smoothie
    else:
        reward_scalar = n_validators / n_smoothie
    total_rewards_ls = [reward * reward_scalar for reward in total_rewards_ls]

    return total_rewards_ls


def plot_kdes_and_sfs():
    years = 5
    trials = 1000
    results_ls = []

    input_ls = [
        (1, None, 'solitarius_1minipool'),
        (3, None, 'solitarius_3minipool'),
        (7, None, 'solitarius_7minipool'),
        (50, None, 'solitarius_50minipool'),
        (1, 3000, '3ksmoothie_any#minipool'),
        # The following all give the same results (assuming trials is large enough)
        # (1, 2000, '2ksmoothie_1'),
        # (5, 2000, '2ksmoothie_5'),
        # (50, 2000, '2ksmoothie_50'),
        # (2000, 2000, '2ksmoothie_50'),
    ]

    for validators, smoothie, label in input_ls:
        res = calc_ppv_ls(
            n_validators=validators,
            n_years=years,
            n_trials=trials,
            n_smoothie=smoothie,
            per_validator=True)
        results_ls.append((res, label))

    # do this first to help select xmax
    # xmin = min([min(resls) for resls in list(zip(*results_ls))[0]])
    # xmax = max([max(resls) for resls in list(zip(*results_ls))[0]])

    xmin, xmax = 0, 30  # setting manually
    xpts = np.linspace(xmin, xmax, num=2000)
    fig, subplots = plt.subplots(2)
    for res, lbl in results_ls:
        ypts = scipy.stats.gaussian_kde(res).evaluate(xpts)
        ypts /= sum(ypts)  # make total area 1
        subplots.flat[0].plot(xpts, ypts, label=lbl)  # kde (~pdf)
        subplots.flat[1].plot(xpts, 1 - np.cumsum(ypts), label=lbl)  # survival function (1-cdf)
    subplots.flat[0].legend()
    subplots.flat[0].set_xlabel('ETH')
    subplots.flat[0].set_ylabel('Probability of this reward')
    subplots.flat[1].legend()
    subplots.flat[1].set_xlabel('ETH')
    subplots.flat[1].set_ylabel('Probability of this\nreward or greater')
    fig.suptitle(f'Per-minipool rewards for various setups after {years} years')
    plt.show()


if __name__ == '__main__':
    # print(calc_ppv_ls(n_validators=10, n_years=1, n_trials=10))

    plot_kdes_and_sfs()
