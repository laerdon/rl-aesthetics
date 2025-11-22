# rl-aesthetics
For CDS onboarding

## process
goal: Can we train a NN to output positions of items on canvas such that they abide by basic aesthetic rules, i.e. not overlapping?

We generate a list of heuristics: We want the elements to not overlap, and to all be left aligned.

These heuristics are used to procedurally generate many samples of "optimal" placements.

The samples, a mapping of the following:

Canvas size (l,w) + List of tuples (l, w) -> List of tuples (x, y)...

corresponding to the placements for each of these objects with the given dimensions. 

After generating the samples, we train our neural network with a loss which takes both the binary satisfaction of our hidden heuristics as well as a loss function calculated by overlap function.

# Contributors
laerdon kim

coolbrother
