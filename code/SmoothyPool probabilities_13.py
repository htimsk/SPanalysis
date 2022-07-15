# Adapted from : https://pintail.xyz/posts/beacon-chain-validator-rewards/
# define annualised base reward (measured in ETH) for n validators
# assuming all validators have an effective balance of 32 ETH
import math
import matplotlib.pyplot as plt
from scipy.stats import binom
import numpy as np
import csv
import pandas as pd
import seaborn as sns
from random import choice, random, seed, gauss
from matplotlib.cm import ScalarMappable 


def loadFlashBotCSV():

    from numpy import genfromtxt
    flashbot_data = genfromtxt('blockReward.csv', delimiter=',')

    #print("This is flashbot_data") # debug line
    #print(flashbot_data) # debug line
    return flashbot_data

##print("Loading the etherscan.io blockRewards data\n")

PPVblocks = loadFlashBotCSV()
cntBlocks = np.shape(PPVblocks)


### Variables 
mcTries = 1 #The number of monte carlo tries to model.
minipools = 1
smoothies = 1999
opTime = 100 # Node Operating Time in years 
#opTime = int(input("Enter the number of years that you intent to validate for: "))
d = 28 # Length of award period in days.
n = 405000 #number of validators from https://beaconcha.in/


### Formulas
t = d / 365.25
periods = int(opTime * 365.25 / d ) # calculare the number of award periods.
slotsValidating = int( d * 24 * 60 * 60 / 12) # slots per award period.
SPparticipants = minipools + smoothies
SPfract = minipools / SPparticipants


print("Monte carlo tries: {mcTries}")
print("Minipools: {minipools}")
print("SPparticipants: {SPparticipants}")
print(f"Years Operating: {opTime}")
print(f"Award periods operating: {periods}")
print(f"Slots validating per award period: {slotsValidating}")


### Functions
def years(x):
    return x * 28/365.25


def avgReturn (column):
    global periods
    global df
    df2=df.query("period == (@periods - 1) ")
    averageReturn = df2[column].mean()
    return averageReturn


def medianReturn (column):
    global periods
    global df
    df2=df.query("period == (@periods - 1) ")
    medianReturn = df2[column].median()
    return medianReturn


def stdReturn (column):
    global periods
    global df
    df2=df.query("period == (@periods - 1) ")
    stdReturn = df2[column].std()
    return stdReturn


def typicalSP ():
    global df
    global periods
    df_SPtypical = pd.DataFrame([], columns=['period', 'avgCutPPVSum'])
    avgCutPPVSum = 0
    for period in range(periods):
        df2=df.query("period == (@period) ")
        avgCutPPVSum = avgCutPPVSum + df2['cutPPV'].mean()
        df_SPtypical.loc[len(df_SPtypical.index)] = [ period, avgCutPPVSum ]
    return df_SPtypical


def calcPPV(v):
    proposals = 0
    blockSum = 0
    blockRewards = np.empty([0])# create empty np array

    for v in range(v):
        ## Random variates enable to simulte real world
        var = binom.rvs(slotsValidating, (1/n), loc=0, size=1) 
        proposals = proposals + sum(var) 

        ## Artifical Modeling
        #var = 2
        #proposals = proposals + var

    for _ in range(proposals):
        ## Random variates enable to simulte real world
        REV = choice(PPVblocks)

        ## Artifical Modeling single tail
##        REV = abs(gauss(0, 1))

        ## Artifical Modeling single tail
        #REV = random()
        
    #    REV = 1

        ## Artifical Modeling - Add lottery blocks
##        lotoOdds = 0.005
##        if random() < lotoOdds:
##            REV = REV + 10000

        blockRewards = np.append(blockRewards, [REV], axis=0).astype(np.float)

        try:
            blockSum = np.sum(blockRewards)
        except ValueError: #Needed when the validaoator is prediced to receive 0 blocks, more likely the shourt the time validating.
            blockSum = 0
    
    return blockSum, proposals

 
def mcTry(tries, minipools, smoothies, df):
    score = 0
    plusScore = 0
    runningScore = 0
    earn = 0
    
    sPPVsum = 0
    singleProposals = 0
    singleProposalsSum = 0

    smoothieProposalsSum = 0
    totalProposalsSum = 0 
    cutPPVSum = 0
    cutProposals = 0
    cutProposalsSum = 0

    for period in range(periods):
        singlePPV, singleProposals = calcPPV(minipools)
        smoothiePPV, smoothieProposals = calcPPV(smoothies)

        spMinis = minipools + smoothies
        spShare = minipools / spMinis


        sPPVsum = sPPVsum + singlePPV
        
        singleProposalsSum = singleProposalsSum + singleProposals
        smoothieProposalsSum = smoothieProposalsSum + smoothieProposals
        totalProposals = singleProposals + smoothieProposals
        totalProposalsSum = totalProposalsSum + totalProposals
        cutProposals = totalProposals * spShare
        cutProposalsSum = cutProposalsSum + cutProposals

        
        totalPPV = singlePPV + smoothiePPV
        cutPPV = totalPPV * spShare
        cutPPVSum = cutPPVSum + cutPPV
        
        
        earn = earn + ((totalPPV * spShare) - singlePPV)
        
        # Evaluate the score on a per award period intervial. If the earn is > in that award period for the SP then +1 to the score
        if (cutPPV > singlePPV): 
            score = 1
            plusScore = 1
            runningScore = runningScore + 1

        if (cutPPV < singlePPV): 
            score = -1

        df.loc[len(df.index)] = [ tries, period, singleProposals, singleProposalsSum, singlePPV/1e18, sPPVsum/1e18, smoothieProposals, smoothieProposalsSum, smoothiePPV/1e18, totalProposals, totalProposalsSum, totalPPV/1e18, spShare, cutProposals, cutProposalsSum, cutPPV/1e18, cutPPVSum/1e18, score, plusScore, runningScore, earn/1e18 ]

        score = 0
        plusScore = 0
         
##        if (period % 10) == 0:
##            print(period)

    return df


def makeDF (minipools, smoothies):
    df = pd.DataFrame([], columns=['tries', 'period', 'sProps', 'sPropsSum', 'sPPV', 'sPPVsum', 'spProps', 'spPropsSum', 'spPPV', 'totalProps', 'totalPropsSum', 'totalPPV', 'spShare', 'cutProps', 'cutPropsSum', 'cutPPV', 'cutPPVSum', 'score', 'plusScore', 'runningScore', 'earn'])

    for tries in range(mcTries):
        df = mcTry(tries, minipools, smoothies, df)
        print(f'This is trie {tries}')

    return df


def performace (minipools, smoothies):
    print(f'Evaluating performance and PROFIT of {minipools} minipools and {smoothies} smoothies.')
    df = makeDF(minipools, smoothies)
    totalPlusScore = df['plusScore'].sum()

    totalPeriods = len(df.index)
    success = totalPlusScore / totalPeriods * 100

    df_Outcome = df[df['period'] == periods - 1 ] # Filters for only final reward period 
    df_wins = df_Outcome[df_Outcome['earn'] > 0] # Selects only the trys were the SP is relative positive
    WLpercent = len(df_wins.index)/mcTries * 100 # Calculates win / loss percent.

    propsGain = ( df_Outcome['cutPropsSum'].sum() / df_Outcome['sPropsSum'].sum() ) - 1
    
    return success, WLpercent, propsGain

##def profit (minipools, smoothies): <---------------- Delete this 
##    
##    df = makeDF(minipools, smoothies)
##
##    
##    return WLpercent

### MAIN PROGRAM


df = makeDF(minipools, smoothies)
#print(df)
df.to_csv('SmoothieAnalysis.csv')

### Some Stats:

print(f"Reward period was {d} days.")
print(f"Staking period was {opTime} years.")
print(f"Total number of beacon chain validators assumed was {n}.")
print(f"Monte carlo tries evaluated was {mcTries}.")
print(f"To generate the bayesian model of PPV:")
print(f' Number of historic blocks sampled was = {cntBlocks[0]}')
print(f' The timespan of historic block rewards sampled was {cntBlocks[0]*12/(60*60*24):.1f} day(s)\n')


avgsPPVsum = avgReturn('sPPVsum')
mediansPPVsum = medianReturn('sPPVsum')
stdsPPVsum = stdReturn('sPPVsum')
textstr_sPPV = '\n'.join((
    r'average = %.2f' % (avgsPPVsum, ),
    r'median = %.2f' % (mediansPPVsum, ),
    r'standard deviation = %.2f' % (stdsPPVsum, )))
print(f'The average, median, std earn at the end of staking Solitarius minipool(s) was {avgsPPVsum:.2f}, {mediansPPVsum:.2f}, +/- {stdsPPVsum:.2f} ETH.')


avgcutPPVSum = avgReturn('cutPPVSum')
mediancutPPVSum = medianReturn('cutPPVSum')
stdcutPPVSum = stdReturn('cutPPVSum')
textstr_cutPPVSum = '\n'.join((
    r'average = %.2f' % (avgcutPPVSum, ),
    r'median = %.2f' % (mediancutPPVSum, ),
    r'standard deviation = %.2f' % (stdcutPPVSum, )))
print(f'The average, median, std earn at the end of staking of SP participating minipool(s) was {avgcutPPVSum:.2f}, {mediancutPPVSum:.2f}, +/- {stdcutPPVSum:.2f} ETH.\n')

### Performacne

tspan = cntBlocks[0]*12/(60*60*24)
totalPlusScore = df['plusScore'].sum()

totalPeriods = len(df.index)
success = totalPlusScore / totalPeriods * 100

textstr_performance = '\n'.join((
    r'Likelihood of SP outperforming = %.1f' % (success, ) + '%',
    r'The success score was +%.0f' % (totalPlusScore, ),
    r'over a total award periods of %.0f' % (totalPeriods, )))
print(f'The likelihood of a SP of {SPparticipants} participating minipools OUTPERFORMING {minipools} Solitarius minipool(s) (f = {SPfract:.2f}) is {success:.1f}%.')
print(f'The results were + {totalPlusScore} successes out of {totalPeriods} award periods.\n')


### Profiting (profiteering) 

df_Outcome = df[df['period'] == periods -1 ]
print(len(df_Outcome.index))
df_wins = df_Outcome[df_Outcome['earn'] > 0]
WLpercent = len(df_wins.index)/mcTries * 100

textstr_WL = '\n'.join((
    r'SP profiting over Solitarius = %.1f' % (WLpercent, ) + '%',
    r'wins = %.0f' % (len(df_wins.index), ),
    r'losses = %.0f' % (mcTries-len(df_wins.index), )))
print(f'The likelihood of the SP of {SPparticipants} participating minipools PROFITING over {minipools} Solitarius minipool(s) (f = {SPfract:.2f}) is {WLpercent:.1f}%.')
print(f'The results were + {len(df_wins.index)} wins out of {mcTries} tries.\n')

### Proposer Gains 

propsGain = ( df_Outcome['cutPropsSum'].sum() / df_Outcome['sPropsSum'].sum() ) - 1
sPropsAll = df_Outcome["sPropsSum"].sum()
cutPropsAll = df_Outcome["cutPropsSum"].sum()

textstr_WL = '\n'.join((
    r'SP proposer gains over Solitarius = %.1f' % (propsGain, ) + '%',
    r'Soitarius proposals = %.0f' % (sPropsAll, ),
    r'SP proposals = %.1f' % (cutPropsAll, )))
print(f'The SP of {SPparticipants} participating minipools had a gain proposing blocks over {minipools} Solitarius minipool(s) (f = {SPfract:.2f}) of {propsGain:.5f}%.')
print(f'The results were {sPropsAll} solitarius vs {cutPropsAll:.1f} share of SP proposals.\n')




## ----------------------------------------
## PLOT FUNCTIONS
## ----------------------------------------

alphaAdjust = alpha=1/math.sqrt(mcTries)
if alphaAdjust < 0.5:
    alphaAdjust = 0.5
    
# Plot smoothing pool runs <<<<<<<<<<<<<<<<<<<<
plt.suptitle('Assuming: ' + str(n) + ' beacon validators; NO operating ' +str(minipools) + ' minipools; ' +str(SPparticipants)+ ' SP participants; f = ' +str(round(SPfract, 3))+ ', ' +str(d)+ ' d reward period; modeled by ' +str(mcTries)+ ' tries.')
for attempt in range(mcTries):
    df2=df.query("tries == @attempt")
    plt.plot(df2[['period']]*28/365, df2[['cutPPVSum']], '-', color='cyan', alpha=1/math.sqrt(mcTries)) # SP Participant
plt.axhline(y=0, color='r', linestyle='dashed', alpha=alphaAdjust)
plt.title('Modeling runs showing commulative ETH earned by participating in the Smooting Pool (SP).')
plt.xlabel('Number of years')
plt.ylabel('ETH')
plt.plot([], label="SP Participant", color="cyan")
plt.legend(loc="upper left")
plt.text(0.05, 0.8, textstr_cutPPVSum, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
figure = plt.gcf() # get current figure
figure.set_size_inches (19.2, 10.8)
plt.savefig('Figure_1.png', dpi = 100)
#plt.show()
plt.close()


# Plot solitary runs <<<<<<<<<<<<<<<<<<<<
plt.suptitle('Assuming: ' + str(n) + ' beacon validators; NO operating ' +str(minipools) + ' minipools; ' +str(SPparticipants)+ ' SP participants; f = ' +str(round(SPfract, 3))+ ', ' +str(d)+ ' d reward period; modeled by ' +str(mcTries)+ ' tries.')
for attempt in range(mcTries):
    df2=df.query("tries == @attempt")
    plt.plot(df2[['period']]*28/365, df2[['sPPVsum']], '-', color='mediumpurple', alpha=alphaAdjust) # Solitarius minipool
plt.axhline(y=0, color='r', linestyle='dashed', alpha=alphaAdjust)
plt.title('Modeling runs showing commulative ETH earned by solitary node operating.')
plt.xlabel('Number of years')
plt.ylabel('ETH')
plt.plot([], label="Solitarius minipool", color="mediumpurple")
plt.legend(loc="upper left")
plt.text(0.05, 0.8, textstr_sPPV, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
figure = plt.gcf() # get current figure
figure.set_size_inches (19.2, 10.8)
plt.savefig('Figure_2.png', dpi = 100)
#plt.show()
plt.close()


# Plot smoothing pool and solitary runs <<<<<<<<<<<<<<<<<<<<
plt.suptitle('Assuming: ' + str(n) + ' beacon validators; NO operating ' +str(minipools) + ' minipools; ' +str(SPparticipants)+ ' SP participants; f = ' +str(round(SPfract, 3))+ ', ' +str(d)+ ' d reward period; modeled by ' +str(mcTries)+ ' tries.')
for attempt in range(mcTries):
    df2=df.query("tries == @attempt")
    plt.plot(df2[['period']]*28/365, df2[['cutPPVSum']], '-', color='cyan', alpha=alphaAdjust) # SP Participant
    plt.plot(df2[['period']]*28/365, df2[['sPPVsum']], '-', color='mediumpurple', alpha=alphaAdjust) # Solitarius minipool
plt.axhline(y=0, color='r', linestyle='dashed', alpha=alphaAdjust)
plt.title('Modeling runs showing Solitary and Smoothing Pool (SP) profit.')
plt.xlabel('Number of years')
plt.ylabel('ETH')
plt.plot([],'-', label="Solitarius minipool", color="mediumpurple")
plt.plot([], '-', label="SP Participant", color="cyan")
plt.legend(loc="upper left")
figure = plt.gcf() # get current figure
figure.set_size_inches (19.2, 10.8)
plt.savefig('Figure_3.png', dpi = 100)
#plt.show()
plt.close()


# Plot TYPICAL SP and solitary runs <<<<<<<<<<<<<<<<<<<<
#####plt.subplot(2, 1, 2)
plt.suptitle('Assuming: ' + str(n) + ' beacon validators; NO operating ' +str(minipools) + ' minipools; ' +str(SPparticipants)+ ' SP participants; f = ' +str(round(SPfract, 3))+ ', ' +str(d)+ ' d reward period; modeled by ' +str(mcTries)+ ' tries.')
for attempt in range(mcTries):
    df2=df.query("tries == @attempt")
    plt.plot(df2[['period']]*28/365, df2[['sPPVsum']], '-', color='mediumpurple', alpha=alphaAdjust) 

df2=df.query("tries == 0")
df_SPtypical = typicalSP()
plt.plot(df_SPtypical[['period']]*28/365, df_SPtypical[['avgCutPPVSum']], '-o', color='cyan')
plt.axhline(y=0, color='r', linestyle='dashed', alpha=alphaAdjust)
plt.title('Modeling runs showing Solitary and averaged Smoothing Pool (SP) profit.')
plt.xlabel('Number of years')
plt.ylabel('ETH')
plt.plot([],'-', label="Solitarius minipool", color="mediumpurple")
plt.plot([], '-', label="SP Participant (avg)", color="cyan")
plt.legend(loc="upper left")
figure = plt.gcf() # get current figure
figure.set_size_inches (19.2, 10.8)
plt.savefig('Figure_4.png', dpi = 100)
#plt.show()
plt.close()


# Plot Net (+/-) ETH earned <<<<<<<<<<<<<<<<<<<<
plt.suptitle('Assuming: ' + str(n) + ' beacon validators; NO operating ' +str(minipools) + ' minipools; ' +str(SPparticipants)+ ' SP participants; f = ' +str(round(SPfract, 3))+ ', ' +str(d)+ ' d reward period; modeled by ' +str(mcTries)+ ' tries.')
#plt.subplot(2, 1, 1)
for attempt in range(mcTries):
    df2=df.query("tries == @attempt")
    plt.plot(df2[['period']], df2[['earn']], '-', color='lightgreen', alpha=alphaAdjust) 
plt.axhline(y=0, color='r', linestyle='dashed', alpha=alphaAdjust)
plt.title('Profit earned by joing the Smoothing Pool (SP) by modeling run.')
plt.xlabel('Number of Award cycles')
plt.ylabel('ETH')
plt.text(0.05, 0.1, textstr_WL, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
figure = plt.gcf() # get current figure
figure.set_size_inches (19.2, 10.8)
plt.savefig('Figure_5.png', dpi = 100)
#plt.show()
plt.close()


# Plot KEYBOARD            <<<<<<<<<<<<<<<<<<<<
# Note: see https://stackoverflow.com/questions/65094280/python-barplot-colored-according-to-a-third-variable


plt.suptitle('Assuming: ' + str(n) + ' beacon validators; NO operating ' +str(minipools) + ' minipools; ' +str(SPparticipants)+ ' SP participants; f = ' +str(round(SPfract, 3))+ ', ' +str(d)+ ' d reward period; modeled by ' +str(mcTries)+ ' tries.')
#plt.subplot(2, 1, 1)
for attempt in range(mcTries):
    df2=df.query("tries == @attempt")
    x_pos = df2[['period']].values.flatten()
    score_simple = df2[['score']].values.flatten()
    mask1 = score_simple > 0
    mask2 = score_simple < 0
    plt.bar(x_pos[mask1], score_simple[mask1], color='darkgreen', alpha=alphaAdjust)
    plt.bar(x_pos[mask2], score_simple[mask2], color='red', alpha=alphaAdjust)
#plt.axhline(y=0, color='r', linestyle='dashed')
plt.title('Score score earned by joining the Smoothing Pool (SP) per reward period; +1 for SP, -1 for Solitarius')
plt.xlabel('Number of award cycles.')
plt.ylabel('Score; Shade intensity is average over the number of tries.')
plt.yticks(np.arange(-1, 2, 1.0))
props = dict(boxstyle='round', facecolor='lightgray')
plt.text(0.91, 0.1, textstr_performance, horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes, bbox=props)
figure = plt.gcf() # get current figure
figure.set_size_inches (19.2, 10.8)
plt.savefig('Figure_6.png', dpi = 100)
#plt.show()
plt.close()

# Plot KEYBOARD2            22222222222222222222222
# Note: see https://stackoverflow.com/questions/65094280/python-barplot-colored-according-to-a-third-variable


plt.suptitle('Assuming: ' + str(n) + ' beacon validators; NO operating ' +str(minipools) + ' minipools; ' +str(SPparticipants)+ ' SP participants; f = ' +str(round(SPfract, 3))+ ', ' +str(d)+ ' d reward period; modeled by ' +str(mcTries)+ ' tries.')
#plt.subplot(2, 1, 1)

df2 = df.groupby('period', as_index =False).mean()
Z = df2[['score']].to_numpy().ravel()
X = df2[['period']].to_numpy().ravel()
Y = np.ones(1)
df_keys = pd.DataFrame([Z], columns=X, index=Y)
print(df_keys)

#sns.barplot(x="period", y=ydata, hue="score", data=df2, palette='siesmic', dodge=False) 
sns.heatmap(df_keys, cmap='coolwarm_r', vmin=-1, vmax=1)


    
#plt.axhline(y=0, color='r', linestyle='dashed')
plt.title('Score score earned by joining the Smoothing Pool (SP) per reward period; +1 for SP, -1 for Solitarius')
plt.xlabel('Number of award cycles.')
plt.ylabel('Score; Shade intensity is average over the number of tries.')
plt.yticks(np.arange(1, 2, 1.0))
plt.xticks(np.arange(0, periods+1, 13))
props = dict(boxstyle='round', facecolor='lightgray')
plt.text(0.95, 0.05, textstr_performance, horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes, bbox=props)
plt.legend().remove()
figure = plt.gcf() # get current figure
figure.set_size_inches (19.2, 10.8)
plt.savefig('Figure_6_2.png', dpi = 100)
#plt.show()
plt.close()



# COLORED MATRIX <<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
x = np.arange(1,10,1) # Create Matrix dimension for number of minipools operated.
#x = [1, 3, 6, 50]
y = np.arange(10,110,10) #SP Participants values 
#y  = [50, 75, 100, 200, 500]
X,Y = np.meshgrid(x, y) # grid of point
print(f' This is the X,Y meshgrid: {X,Y}')

Z = np.zeros((len(y),len(x))) # Note the flip in coordinate to get the dimension correct
Zp = np.zeros((len(y),len(x))) # Note the flip in coordinate to get the dimension correct
Zg = np.zeros((len(y),len(x))) # Note the flip in coordinate to get the dimension correct
Zfract_df = pd.DataFrame([], columns=['fraction', 'success', 'WL', 'propsGain', 'nMinis'])


for i in range(len(x)):
    xi = x[i]
    for j in range(len(y)):
        yj = y[j]
        #print(f'i = {i}')
        #print(f'xi = {xi}')
        #print(f'j = {j}')
        #print(f'yj = {yj}')
        z, zp, zg = performace(xi, (yj-xi)) # evaluation of the function on the grid
        #print(f'z = {z}')
        Z[j,i]=z/100
        Zp[j,i]=zp/100
        Zg[j,i]=zg

        fraction = xi/yj

        success = z/100
        WL = zp/100
        propsGain = zg

        Zfract_df.loc[len(Zfract_df.index)] = [ fraction, success, WL, propsGain, xi ] 

print(f'Udataed Z = {Z}')
print(f'Udataed Zp = {Zp}')
print(f'Udataed Zg = {Zg}')
print(f'Zfract_df = {Zfract_df}')
Zfract_df.to_csv('Zfract_df.csv')

# Create a X,Y plot
#xLabels = ['1', '3', '6', '50']
xLabels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
#yLabels = ['50', '75', '100', '200', '500']
yLabels = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

fig, ax = plt.subplots()
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

#p1 = plt.imshow(Z, origin="lower", cmap='RdYlGn') # Adjust vmina and vmax for the color spectrum
p1 = sns.heatmap(Z, cmap='RdYlGn', annot=True, center=0.58, fmt ='.0%') # Adjust vmina and vmax for the color spectrum use center=0.58 for RdYlGn at 66%
#plt.colorbar()

plt.suptitle('Assuming: ' + str(n) + ' beacon validators; staking for ' + str(opTime) +' year(s); ' +str(d)+ ' d reward period; modeled by ' +str(mcTries)+ ' tries.')
plt.title('Likelihood of Smoothing Pool (SP) participation outperforming Solitary mode.', fontsize=12)

plt.xlabel("Solitarius minipools.")
##ax.set_xticks(range(len(x)))
##ax.set_xticklabels(xLabels)
#plt.margins(x=0, y=0)
plt.xticks(np.arange(len(xLabels)) + 0.5, xLabels)


plt.ylabel("SP minipools (inclusive of NO minipools).")
##ax.set_yticks(range(len(y)))
##ax.set_yticklabels(yLabels)
plt.yticks(np.arange(len(yLabels)) + 0.5, yLabels)

plt.xlim([0, len(x)]) # Set range of x axis here Need scale....
plt.ylim([0, len(y)]) # Set range of y axis here
figure = plt.gcf() # get current figure
figure.set_size_inches (19.2, 10.8)
plt.savefig('Figure_7.1.png', dpi = 100)

#Add contours
#CS = plt.contour(Z, levels=[.50, .667 ], colors=['#000000'], extend='both') #

##fmt = {}
##strs = ['50%', '66.7%'] #
##for l, s in zip(CS.levels, strs):
##    fmt[l] = s
##    
##plt.clabel((CS), inline=True, manual=True, fmt=fmt, fontsize=8)

#plt.show()
plt.close()


# PROFIT MATRIX Chart
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

#p1 = plt.imshow(Z, origin="lower", cmap='RdYlGn') # Adjust vmina and vmax for the color spectrum
p1 = sns.heatmap(Zp, cmap='RdYlGn', annot=True, center=0.58, fmt ='.0%') # Adjust vmina and vmax for the color spectrum use center=0.58 for RdYlGn at 66%
#plt.colorbar()

plt.suptitle('Assuming: ' + str(n) + ' beacon validators; staking for ' + str(opTime) +' year(s); ' +str(d)+ ' d reward period; modeled by ' +str(mcTries)+ ' tries.')
plt.title('Likelihood of Smooting Pool (SP) participation PROFITING over Solitary mode.', fontsize=12)

plt.xlabel("Solitarius minipools.")
##ax.set_xticks(range(len(x)))
##ax.set_xticklabels(xLabels)
#plt.margins(x=0, y=0)
plt.xticks(np.arange(len(xLabels)) + 0.5, xLabels)


plt.ylabel("SP minipools (inclusive of NO minipools).")
##ax.set_yticks(range(len(y)))
##ax.set_yticklabels(yLabels)
plt.yticks(np.arange(len(yLabels)) + 0.5, yLabels)

plt.xlim([0, len(x)]) # Set range of x axis here Need scale....
plt.ylim([0, len(y)]) # Set range of y axis here
figure = plt.gcf() # get current figure
figure.set_size_inches (19.2, 10.8)
plt.savefig('Figure_7.2.png', dpi = 100)
#plt.show()
plt.close()


### Plot Zfract_df <<<<<<<<<<<<<<<<<<<<
### ----------------------------------------------------------------
plt.suptitle('Assuming: ' + str(n) + ' beacon validators; staking for ' + str(opTime) +' year(s); ' +str(d)+ ' d reward period; modeled by ' +str(mcTries)+ ' tries.')
plt.title('Likelihood of Smoothing Pool (SP) participation out performing Solitary mode measured at the end of each ' +str(d)+ ' d reward period.', fontsize=12)

ax = sns.regplot(x="fraction", y="success", data=Zfract_df, order=3, scatter=False)
sns.scatterplot(x="fraction", y="success", data=Zfract_df, hue='nMinis', palette="tab10", legend=False)
plt.xlabel('Fraction (minipools / SP minipool participants)')
plt.ylabel('Success Rate')
#plt.plot([], label="Fit", color="darkblue")

plt.axhline(y=(.5), color='grey', linestyle='dashed', label='Common Shares Expected Performance')
plt.axhline(y=(2/3), color='g', linestyle='dotted', alpha=0.5, label='67% confidence level')
#plt.text(0.05, 0.8, textstr_cutPPVSum, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
plt.xlim([0, 1]) # Set range of x axis here Need scale....
plt.ylim([0, 1]) # Set range of y axis here
plt.legend(loc="upper left")
figure = plt.gcf() # get current figure
figure.set_size_inches (19.2, 10.8)
plt.savefig('Figure_8.1.png', dpi = 100)
#plt.show()
plt.close()


### Plot Zfract_df <<<<<<<<<<<<<<<<<<<<
### ----------------------------------------------------------------
plt.suptitle('Assuming: ' + str(n) + ' beacon validators; staking for ' + str(opTime) +' year(s); ' +str(d)+ ' d reward period; modeled by ' +str(mcTries)+ ' tries.')
plt.title('Likelihood of Smoothing Pool (SP) participation PROFITING over Solitary mode at the end ' + str(opTime) +' year(s).', fontsize=12)


ax = sns.regplot(x="fraction", y="WL", data=Zfract_df, order=3, scatter=False, color='black',)
sns.scatterplot(x="fraction", y="WL", data=Zfract_df, hue='nMinis', palette="flag", legend=False)
plt.xlabel('Fraction (minipools / SP minipool participants)')
plt.ylabel('Win:Loss Ratio')
#plt.plot([], label="Fit", color="darkblue")

plt.axhline(y=(.5), color='grey', linestyle='dashed', label='Common Shares Expected Performance')
plt.axhline(y=(2/3), color='g', linestyle='dotted', alpha=0.5, label='67% confidence level')
#plt.text(0.05, 0.8, textstr_cutPPVSum, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
plt.xlim([0, 1]) # Set range of x axis here Need scale....
plt.ylim([0, 1]) # Set range of y axis here
plt.legend(loc="upper left")
figure = plt.gcf() # get current figure
figure.set_size_inches (19.2, 10.8)
plt.savefig('Figure_8.2.png', dpi = 100)
#plt.show()
plt.close()



# Proposals MATRIX Chart
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

upper = np.amax(Zg)
lower = np.amin(Zg)
rangeLimit = max(abs(upper), abs(lower))

#p1 = plt.imshow(Z, origin="lower", cmap='RdYlGn') # Adjust vmina and vmax for the color spectrum
p1 = sns.heatmap(Zg, cmap='RdBu', annot=True, fmt ='.1%', vmin=-rangeLimit, vmax=rangeLimit) # Adjust vmina and vmax for the color spectrum use for RdYlGn
#plt.colorbar()

plt.suptitle('Assuming: ' + str(n) + ' beacon validators; staking for ' + str(opTime) +' year(s); ' +str(d)+ ' d reward period; modeled by ' +str(mcTries)+ ' tries.')
plt.title('Proposal Gain of Smooting Pool (SP) participation over Solitary mode.', fontsize=12)

plt.xlabel("Solitarius minipools.")
##ax.set_xticks(range(len(x)))
##ax.set_xticklabels(xLabels)
#plt.margins(x=0, y=0)
plt.xticks(np.arange(len(xLabels)) + 0.5, xLabels)


plt.ylabel("SP minipools (inclusive of NO minipools).")
##ax.set_yticks(range(len(y)))
##ax.set_yticklabels(yLabels)
plt.yticks(np.arange(len(yLabels)) + 0.5, yLabels)

plt.xlim([0, len(x)]) # Set range of x axis here Need scale....
plt.ylim([0, len(y)]) # Set range of y axis here
figure = plt.gcf() # get current figure
figure.set_size_inches (19.2, 10.8)
plt.savefig('Figure_7.3.png', dpi = 100)
#plt.show()
plt.close()

### Plot Proposals <<<<<<<<<<<<<<<<<<<<
### ----------------------------------------------------------------

plt.suptitle('Assuming: ' + str(n) + ' beacon validators; staking for ' + str(opTime) +' year(s); ' +str(d)+ ' d reward period; modeled by ' +str(mcTries)+ ' tries.')
plt.title('Proposal Gain of Smoothing Pool (SP) participation out performing Solitary mode measured at the end of each ' +str(d)+ ' d reward period.', fontsize=12)


ax = sns.regplot(x="fraction", y="propsGain", color='orange', data=Zfract_df, order=3, scatter=False)
sns.scatterplot(x="fraction", y="propsGain", data=Zfract_df, hue='nMinis', palette="tab10", legend=False)
plt.xlabel('Fraction (minipools / SP minipool participants)')
plt.ylabel('Gain Rate')
#plt.plot([], label="Fit", color="darkblue")

plt.axhline(y=(.5), color='grey', linestyle='dashed', label='Common Shares Expected Performance')
#plt.axhline(y=(2/3), color='g', linestyle='dotted', alpha=0.5, label='67% confidence level')
#plt.text(0.05, 0.8, textstr_cutPPVSum, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
plt.xlim([0, 1]) # Set range of x axis here Need scale....
plt.ylim([0.2, -0.2]) # Set range of y axis here
plt.legend(loc="upper left")
figure = plt.gcf() # get current figure
figure.set_size_inches (19.2, 10.8)
plt.savefig('Figure_8.3.png', dpi = 100)
#plt.show()
plt.close()

print('End of Program')
