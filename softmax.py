# Eli5 Softmax
# https://www.reddit.com/r/MachineLearning/comments/42ny98/eli5_the_softmax_algorithm/?
# Rescales the values so that they sum to one

import numpy as np

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x),axis=0)

values = [3,1.2,0.55]
rescaled = softmax(values)
print(rescaled)
print("which should sum to:")
print(np.sum(rescaled))



# FuschiaKnight
# Let's say you have some scores for how strongly you feel between three choices. For instance, you're looking for your phone, which you've left in either your kitchen, bedroom, or bathroom. You've lost your phone 7 times before: 4 in the bedroom, 2 in the kitchen, and once in the bathroom. If you lose your phone and want to compute how likely it is to be found in each room (based on your past experience), then the answer is simply
# P(bedroom) = 4 / (4+2+1) = 4/7
# P(kitchen) = 2 / (4+2+1) = 2/7
# P(bathroom) = 1 / (4+2+1) = 1/7
#
# This strategy works pretty well, each numerator is the score and the denominator is the sum of every score so that the terms are now normalized to sum to 1 (as any probability distribution must).
# Okay, cool.
# But what happens if these scores can be negative numbers. For instance, you're not really confident where you left your phone. You estimate, it's likely it's in your room (so give it like a 4), you're entirely unsure if it's in the kitchen (give it a 0), and you are pretty confidence it's not likely in the bathroom (give it a -1).
# Well now there's a problem because you end up with negative probabilities, which is weird. What does it mean to assign a probability of -1/3 to an event? Clearly this makes no sense.
# So what do we do? We do something to guarantee that every term is positive. So we replace each term x with ex. Now even negative numbers become positive, and obviously the bigger the original number, the bigger the new number. Cool!
# Now we have
# P(bedroom) = exp(4) / (exp(4)+exp(0)+exp(-1)) = 0.976
# P(kitchen) = exp(0) / (exp(4)+exp(0)+exp(-1)) = 0.018
# P(bathroom) = exp(-1) / (exp(4)+exp(0)+exp(-1)) = 0.007
# And all is right with the world!
# For your example, the softmax operation applied to your vector is

def softmax_v2(x):
	positives = [np.exp(i) for i in x]
	normalised_denom = np.sum(positives)
	return([p/normalised_denom for p in positives])

print("Version 2")
print(softmax_v2(values))
