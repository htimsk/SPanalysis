from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import binom
import pandas as pd
import math

### Variables 
Aminipools = 1
Bminipools = 200

d = 365 * 5 # Length of award period in days.
n = 425000 # Number of validators from https://beaconcha.in/


### Formulas
#t = d / 365.25
slotsValidating = int( d * 24 * 60 * 60 / 12) # slots per award period.
xrange = np.arange(0, 500000, 1, dtype=int) # Could be something like myRange = range(1,1000,1)


## Functions
factor = Aminipools / Bminipools

bDataA = binom.pmf(xrange, slotsValidating, (Aminipools/n))
bCDF_A = binom.cdf(xrange, slotsValidating, (Aminipools/n))
bDataB = binom.pmf(xrange, slotsValidating, (Bminipools/n))
bCDF_B = binom.cdf(xrange, slotsValidating, (Bminipools/n))
Amean, Avar, Askew, Akurt = binom.stats(slotsValidating, (Aminipools/n), moments='mvsk')
Bmean, Bvar, Bskew, Bkurt = binom.stats(slotsValidating, (Bminipools/n), moments='mvsk')

selected = int(Amean)


below=[]
on=[]
above=[]
totals=[]
losses = []
pushes = []
wins = []
LPWtotal = []

subset = 1000

for x in range(subset):  
    below_i = bCDF_B[int(math.ceil(x / factor)) - 1] #subtract 1 to get below the ith bin. 
    if x == 0:
        below_i = 0
##    print('x', x)
##    print('x / factor', x / factor)
##    print('int(x / factor)', int(math.ceil(x / factor)))
    on_i = bDataB[int(math.ceil(x / factor))]
    above_i = 1 - bCDF_B[int(math.ceil(x / factor))]
    totals_i = below_i + on_i + above_i

    losses_i = bDataA[int(x)] * below_i
    pushes_i = bDataA[int(x)] * on_i
    wins_i = bDataA[int(x)] * above_i
    LPWtotal_i = losses_i + pushes_i + wins_i

    below.append(below_i)
    on.append(on_i)
    above.append(above_i)
    totals.append(totals_i)
    losses.append(losses_i)
    pushes.append(pushes_i)
    wins.append(wins_i)
    LPWtotal.append(LPWtotal_i)

df = pd.DataFrame({'x': xrange,
                   'bDataA'  :pd.Series(bDataA),
                   'bCDF_A'  :pd.Series(bCDF_A),
                   'bDataB'  :pd.Series(bDataB),
                   'bCDF_B'  :pd.Series(bCDF_B),
                   'below'   :pd.Series(below),
                   'on'      :pd.Series(on),
                   'above'   :pd.Series(above),
                   'totals'  :pd.Series(totals),
                   'losses'  :pd.Series(losses),
                   'pushes'  :pd.Series(pushes),
                   'wins'    :pd.Series(wins),
                   'LPWtotal':pd.Series(LPWtotal),
                   })
print(df)
df.to_csv('pdfData.csv')

totalLosses = df['losses'].sum(axis=0)
totalPushes = df['pushes'].sum(axis=0)
totalWins = df['wins'].sum(axis=0)
spAdvantage = ((totalWins / totalLosses) - 1 ) * 100
errorCheck = totalLosses + totalPushes + totalWins



#print(f"Years Operating: {opTime}")
print(f"Reward period was {d} days.")
#print(f"Award periods operating: {periods}")
print(f"Total number of beacon chain validators assumed was {n}.")
print(f"Slots validating per award period: {slotsValidating}\n")

print(f"Minipools: {Aminipools}")
print(f"SP participants: {Bminipools}\n")
print('f fraction  : ', factor)
print()


print('4 moments of the binomial distribution of the Solitarius set of minipools (A):')
print(f'Amean: {Amean:.9f}')
print(f'Avar : {Avar:.9f}')
print(f'Askew: {Askew:.9f}')
print(f'Akurt: {Akurt:.9f}')
print()


print('4 moments of the binomial distribution of the SP minipools (B):')
print(f'Bmean: {Bmean:.9f}')
print(f'Bvar : {Bvar:.9f}')
print(f'Bskew: {Bskew:.9f}')
print(f'Bkurt: {Bkurt:.9f}')
print()

print('4 moments of the binomial distribution normalized:')
print(f'ABmean/mp: {Amean/Aminipools:.9f} Bmean/mp : {Bmean/Bminipools:.9f}')
print(f'Avar/mp  : { Avar/Aminipools:.9f} Bvar/mp  : {Bvar/Bminipools:.9f}')
print(f'Askew/mp : {Askew/Aminipools:.9f} Bskew/mp : {Bskew/Bminipools:.9f}')
print(f'Akurt/mp : {Akurt/Aminipools:.9f} Bkurt/mp : {Bkurt/Bminipools:.9f}')
print()


print(f'errorCheck   : {errorCheck:.9f}')

print('Smoothing Pool Stats:')
print(f'      Losses: {totalLosses:.3f}')
print(f'      Pushes: {totalPushes:.3f}')
print(f'        Wins: {totalWins:.3f}')
print()
print(f'SP advantage: {spAdvantage:.2f}%')




clrsA = ['blue' if x == selected
        else 'lightblue' 
        for x in xrange]

clrsB = ['grey' if x == selected / factor
        else 'red' if x < selected / factor
        else 'green'
        for x in xrange]

df2 = df.head(subset)

##
fig, ax = plt.subplots()
##
axA = sns.barplot(x='x', y='bDataA', data=df2, label=str(Aminipools) + ' minipool(s)', palette=clrsA)
#axB = sns.barplot(x='x', y='bDataB', data=df, label=str(Bminipools) + ' minipool(s)', palette=clrsB, alpha = 0.5)
#sns.lineplot(x='x', y='bCDF_A', data=df, label=str(Aminipools) + ' minipool(s)', palette=clrsA)

plt.xticks(np.arange(0, 500, step=2))
plt.xlim([-1, 100]) # Set range of x axis here Need scale....
plt.legend(loc="upper right")

# label each bar in histogram
for p in ax.patches: 
 height = p.get_height() # get the height of each bar
 # adding text to each bar
 if height > 0.001: 
     ax.text(x = p.get_x()+(p.get_width()/2), # x-coordinate position of data label, padded to be in the middle of the bar
     fontsize=6,
     y = height+0.002, # y-coordinate position of data label, padded 0.2 above bar
     s = '{:.3f}'.format(height), # data label, formatted to ignore decimals
     ha = 'center') # sets horizontal alignment (ha) to center


figure = plt.gcf() # get current figure
figure.set_size_inches (19.2, 10.8)
plt.savefig('Figure_1of6.png', dpi = 100)
plt.show()
plt.close()


