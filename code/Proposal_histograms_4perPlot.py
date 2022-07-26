
# Adapted from : https://pintail.xyz/posts/beacon-chain-validator-rewards/

# define annualised base reward (measured in ETH) for n validators
# assuming all validators have an effective balance of 32 ETH

# 5.9.2022 This just takes an average MEV/bock and makes an estimate using the binomial
# probability. It does not take into account the high varability or MEV blocks.
import math
import matplotlib.pyplot as plt
from scipy.stats import binom
import numpy as np
import random
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binom
import pandas as pd


x = [el for el in range(5000)] ## <--- need to adjust



### Variables
minipools = [1, 3, 7, 50]
opTime = 5
d = 28  # Length of award period in days.
n = 425000 #number of validators from https://beaconcha.in/

### Formulas
t = d / 365.25
periods = int(opTime * 365.25 / d ) # calculare the number of award periods.
slotsValidating = int( d * 24 * 60 * 60 / 12) # slots per award period.
print(f"Years Operating: {opTime}")
print(f"Award periods operating: {periods}")
print(f"Slots validating per award period: {slotsValidating}")

# X is the chart x input number of propsal oppurtunities a year
# Y is the pmf (probability mass function) - number of propsal opportunites per year.

def pmf (mini):
    print(f'mini passed to pmf is : {mini}')
    y = binom.pmf(x, slotsValidating, (mini/n))
    return y



#### ====================================================================
#### plot 0 - In color all overlaping on same axis
#### ====================================================================

##fig, ax = plt.subplots()
##
##for mini in minipools :
##    y = pmf(mini)
##    ax.bar(x, y, label=str(mini) + ' minipool(s)', alpha=0.5)
##
##ax.set_xlim(xmin=0)
##ax.set_xlim(xmax=20)
##plt.xticks(range(1, 50))
##ax.set_ylim(ymin=0)
##plt.suptitle('Probability mass function per ' + str(d) + ' days(s)')
##ax.set_title('')
##ax.set_xlabel('Number of block proposal opportunities in ' + str(d) + ' day(s)')
##ax.set_ylabel('Probability')
##plt.legend()
##plt.show()
##plt.close()



#### ====================================================================
#### plot 1 4 plot format
#### ====================================================================
plt.suptitle('Assuming: ' + str(n) + ' beacon validators.')
label0 = str(minipools[0]) + " minipool(s)"
label1 = str(minipools[1]) + " minipool(s)"
label2 = str(minipools[2]) + " minipool(s)"
label3 = str(minipools[3]) + " minipool(s)"


fig, ax = plt.subplots(2, 2, sharey=False, sharex=False)
# add a big axes, hide frame
fig.add_subplot(111, frameon=False)


ax[0,0].bar(x, pmf(minipools[0]), alpha=0.7)
ax[0,1].bar(x, pmf(minipools[1]), alpha=0.7)
ax[1,0].bar(x, pmf(minipools[2]), alpha=0.7)
ax[1,1].bar(x, pmf(minipools[3]), alpha=0.7)

ax[0,0].set_xlim(xmin=0, xmax=10)
#ax[0,0].set_xlim(xmax=10)
ax[0,0].set_ylim(ymin=0, ymax=0.7)
ax[0,0].set_title(label0)

ax[0,1].set_xlim(xmin=0, xmax=10)
#ax[0,1].set_xlim(xmax=10)
ax[0,1].set_ylim(ymin=0, ymax=0.7)
ax[0,1].set_title(label1)

ax[1,0].set_xlim(xmin=0, xmax=40)
#ax[1,0].set_xlim(xmax=20)
ax[1,0].set_ylim(ymin=0, ymax=0.2)
ax[1,0].set_title(label2)

ax[1,1].set_xlim(xmin=0, xmax=40)
#ax[1,1].set_xlim(xmax=60)
ax[1,1].set_ylim(ymin=0, ymax=0.2)
ax[1,1].set_title(label3)


plt.suptitle('Probability mass function for an award period of ' + str(d) + ' days(s).')
plt.xticks(range(1, 50, 5))
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel('Number of block proposal opportunities.')
plt.ylabel('Probability', labelpad=20)
figure = plt.gcf() # get current figure
figure.set_size_inches (19.2, 10.8)
plt.savefig('Figure_1.png', dpi = 100)
#plt.show()
plt.close()

#### ====================================================================
#### plot 2
#### ====================================================================
plt.suptitle('Assuming: ' + str(n) + ' beacon validators.')
d = opTime * 365.25
slotsValidating = int( d * 24 * 60 * 60 / 12) # slots per award period.

label0 = str(minipools[0]) + " minipool(s)"
label1 = str(minipools[1]) + " minipool(s)"
label2 = str(minipools[2]) + " minipool(s)"
label3 = str(minipools[3]) + " minipool(s)"


fig, ax = plt.subplots(2, 2, sharey=False, sharex=False)
# add a big axes, hide frame
fig.add_subplot(111, frameon=False)


ax[0,0].bar(x, pmf(minipools[0]), alpha=0.7)
ax[0,1].bar(x, pmf(minipools[1]), alpha=0.7)
ax[1,0].bar(x, pmf(minipools[2]), alpha=0.7)
ax[1,1].bar(x, pmf(minipools[3]), alpha=0.7)

ax[0,0].set_xlim(xmin=0, xmax=150)
#ax[0,0].set_xlim(xmax=10)
#ax[0,0].set_ylim(ymin=0, ymax=0.07)
ax[0,0].set_title(label0)

ax[0,1].set_xlim(xmin=0, xmax=150)
#ax[0,1].set_xlim(xmax=10)
#ax[0,1].set_ylim(ymin=0, ymax=0.07)
ax[0,1].set_title(label1)

ax[1,0].set_xlim(xmin=150, xmax=300)
#ax[1,0].set_xlim(xmax=20)
#ax[1,0].set_ylim(ymin=0, ymax=0.02)
ax[1,0].set_title(label2)

ax[1,1].set_xlim(xmin=1400, xmax=1700)
#ax[1,1].set_xlim(xmax=60)
#ax[1,1].set_ylim(ymin=0, ymax=0.2)
ax[1,1].set_title(label3)


plt.suptitle('Probability mass function for an validating period of ' + str(opTime) + ' year(s).')
plt.xticks(range(1, 50, 5))
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel('Number of block proposal opportunities.')
plt.ylabel('Probability', labelpad=20)
figure = plt.gcf() # get current figure
figure.set_size_inches (19.2, 10.8)
plt.savefig('Figure_2.png', dpi = 100)
#plt.show()
plt.close()

#### ====================================================================
#### Colorized Version of plot 3
#### ====================================================================
##label0 = str(minipools[0]) + "minipool(s)"
##label1 = str(minipools[1]) + "minipool(s)"
##label2 = str(minipools[2]) + "minipool(s)"
##label3 = str(minipools[3]) + "minipool(s)"
##
##
##fig, ax = plt.subplots(2, 2, sharey=False, sharex=False)
### add a big axes, hide frame
##fig.add_subplot(111, frameon=False)
##
##
##ax[0,0].bar(x, pmf3(minipools[0]), color='red', alpha=0.7)
##ax[0,1].bar(x, pmf3(minipools[1]), color='blue', alpha=0.7)
##ax[1,0].bar(x, pmf3(minipools[2]), color='green', alpha=0.7)
##ax[1,1].bar(x, pmf3(minipools[3]), color='purple', alpha=0.7)
##
##
##ax[0,0].set_xlim(xmin=0, xmax=10)
###ax[0,0].set_xlim(xmax=10)
##ax[0,0].set_ylim(ymin=0, ymax=0.7)
##ax[0,0].set_title(label0)
##
##ax[0,1].set_xlim(xmin=0, xmax=10)
###ax[0,1].set_xlim(xmax=10)
##ax[0,1].set_ylim(ymin=0, ymax=0.7)
##ax[0,1].set_title(label1)
##
##ax[1,0].set_xlim(xmin=0, xmax=40)
###ax[1,0].set_xlim(xmax=20)
##ax[1,0].set_ylim(ymin=0, ymax=0.2)
##ax[1,0].set_title(label2)
##
##ax[1,1].set_xlim(xmin=0, xmax=40)
###ax[1,1].set_xlim(xmax=60)
##ax[1,1].set_ylim(ymin=0, ymax=0.2)
##ax[1,1].set_title(label3)
##
##
##plt.suptitle('Probability mass function for an award period' + str(d) + ' days(s).')
##plt.xticks(range(1, 50, 5))
##plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
##plt.grid(False)
##plt.xlabel('Number of block proposal opportunities.')
##plt.ylabel('Probability', labelpad=5)
##figure = plt.gcf() # get current figure
##figure.set_size_inches (19.2, 10.8)
##plt.savefig('Figure_1c.png', dpi = 100)
##plt.show()
##plt.close()




print("End of Program")


