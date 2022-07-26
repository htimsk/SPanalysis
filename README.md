# Modeling the Profitability of the Rocket Pool Smoothing Pool

This Monte Carlo based analysis attempts to quantify the likely real-world performance of a Rocket Pool (RP) minipool (i.e., an ethereum validator) that joins the smoothing pool (SP) versus remaining a *solitarius* minipool.  The SP is an opt-in feature that will collectively pool the PPV of every member opted into it.  *Solitarius* minipools are RP validators whose node operators (NO) have chosen not to opt-in to the SP and remain in their solitary configuration.

We accomplished the modeling using historical Ethereum proof-of-work mining block rewards to calculate future Ethereum staking (beacon chain) proposer payment value (PPV) rewards.  We used an estimate of the number of beacon chain validators to determine the likelihood that a given minipool would be selected to propose a block post-merge.

Many Monte Carlo tries were performed, predicting the profitability of a set of two cohorts: a collection of solitary minipool(s) and the same set of minipools operated as participants in the SP.  Profitability was defined as the amount of PPV earned in a fixed time interval.

We generated a series of plots that displayed the performance of both cohorts over time.  We then calculated an average win-loss ratio over all the Monte Carlo tries to determine if joining the SP provided a performance advantage.  Finally, we repeated the simulation for four configurations of minipools (1, 3, 10, and 50) operated by a single NO. 

We also compiled a series of heat maps looking at a combination of ratios of NO minipools to the number of SP participants to determine how the performance varies based on the relative sizes of the two sets.  We calculated an *f* fraction representing a NO proportion of the minipool in the SP and plotted that against the performance advantage. 

**Tl;dr A NO participating in the SP is more likely to receive larger monthly ETH rewards than running solitarius minipools.  This depends on two prerequisites.  First, the NO joins the SP only when their minipools will not compose the majority of minipools in the SP.  Second, the NO will not validate for an indefinite time.**

[Download a pdf of the report](https://github.com/htimsk/SPanalysis/raw/main/report/Analysis%20of%20the%20Smoothing%20Pool.pdf)

[Google docs version for commenting](https://docs.google.com/document/d/1dTYbES2mypo06R7Bd1LOzYpGnkHyU0TVWd6Vat9HIGI/edit?usp=sharing)