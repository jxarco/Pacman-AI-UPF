# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(iterations):

          updatedValues = self.values.copy()

          # For each state choose the max Qvalue
          for state in self.mdp.getStates():
            v_max = None
            for action in self.mdp.getPossibleActions(state):
              v = self.getQValue(state, action)
              v_max = max( v, v_max )
            if self.mdp.isTerminal(state):
              v_max = 0

            # This contains the max Qvalues of each state
            updatedValues[state] = v_max
          self.values = updatedValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Qvalue(s,a) = [Sum(Pa(s'|s)*[r(s,a) + (discount * factorVi(s'))]

        Qvalue = 0

        # Esta nos da el estado y la pa.
        states = self.mdp.getTransitionStatesAndProbs(state, action)

        # Apply the Value iteration formula
        for nextState, pa in states:
          reward = self.mdp.getReward(state, action, nextState)
          Qvalue +=  pa * (reward + (self.discount * self.values[nextState]))

        return Qvalue

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        actions = self.mdp.getPossibleActions(state)

        if len(actions) == 0:
          return None

        # Choose the best action

        Qvalues = [self.getQValue(state, action) for action in actions]
        bestQvalue = max(Qvalues)
        bestQIndex = [index for index in range(len(Qvalues)) if Qvalues[index] == bestQvalue]

        # bestQIndex contains the max values: could be more than one, so:

        # We get the first
        index = bestQIndex[0]
          
        return actions[index]

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
