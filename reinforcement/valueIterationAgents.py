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
import collections

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
        
        # self.count = 1

        self.runValueIteration()
        # added code      

    def runValueIteration(self):
        # Write value iteration code here

        for i in range(self.iterations):
        # while self.count <= self.iterations:

          changedValues = util.Counter()

          # create temporary changed values to store and read prev values from getValue
          for s in self.mdp.getStates():

            if self.mdp.isTerminal(s):
              changedValues[s] = 0
            else:
              changedValues[s] = max([self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)])

          # self.count += 1

          # change 
          self.values = changedValues


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
        val = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
          #if self.mdp.isTerminal(nextState):
          #  val += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.prevValues[state])
          #else:

          # change prevValues to getValue
          val += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState))
        return val 
        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestVal = float("-inf")
        bestAction = None
        for action in self.mdp.getPossibleActions(state):
          val = self.computeQValueFromValues(state, action)
          if val > bestVal:
            bestVal = val
            bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        # print "hi: ", self.computeQValueFromValues(state, action)
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        # self.index = -1
        self.statesList = mdp.getStates()
        # print("self.statesList is " + str(self.statesList))
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        
        for i in range(self.iterations):

          s = self.statesList[i % len(self.statesList)]

          if self.mdp.isTerminal(s):
            self.values[s] = 0
          else:
            self.values[s] = max([self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)])


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Compute predecessors of all states
        # dictionary key = state : value = set of predecessors
        predecessors = {key: set() for key in self.mdp.getStates()}

        for s in self.mdp.getStates():
          for action in self.mdp.getPossibleActions(s):
            for nextState, probability in self.mdp.getTransitionStatesAndProbs(s, action):
              if probability > 0.0:
                predecessors[nextState].add(s)

        # Initialize an empty priority queue
        pq = util.PriorityQueue()

        for s in self.mdp.getStates():
          if self.mdp.isTerminal(s):
            continue
          if not self.mdp.isTerminal(s):
            diff = abs(self.values[s] - max([self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)]))
            pq.update(s, -diff)

        for i in range(self.iterations):
          if pq.isEmpty():
            return
          s = pq.pop()

          if not self.mdp.isTerminal(s):
            self.values[s] = self.getQValue(s, self.getAction(s))

            for p in predecessors[s]:
              if not self.mdp.isTerminal(p):
                
                maxQ = float("-inf")
                for action in self.mdp.getPossibleActions(p):
                  qval = self.getQValue(p, action)
                  if qval > maxQ:
                    maxQ = qval
                diff = abs(self.values[p] - maxQ)

              if diff > self.theta:
                pq.update(p, -diff)
                

                
