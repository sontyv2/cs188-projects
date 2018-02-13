# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        print("out of " + str(scores) + " the best score is " + str(bestScore))
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        print("chosen idx", legalMoves[chosenIndex])

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        print('NEWFOOD')
        print(newFood)

        manhatGhost = float("inf")
        food_count = successorGameState.getNumFood()
        print("food count: ", food_count)

        ghostList = [s.getPosition() for s in newGhostStates]

        for ghost in ghostList:
          # manhatGhost = sum of ghost distances
          # manhatGhost += (abs(newPos[0] - ghost[0]) + abs(newPos[1] - ghost[1]))

          # manhatGhost = closest ghost distance
          manhatGhost = min(manhatGhost, abs(newPos[0] - ghost[0]) + abs(newPos[1] - ghost[1]))

        if (manhatGhost <= 2):
          return 0

        # find nearest food
        manhatFood = float("inf")
        # for x in len(newFood):
        #   for y in len(newFood[0]):
        for index in newFood.asList():
          x, y = index
          manhatFood = min(manhatFood, abs(newPos[0] - x) + abs(newPos[1] - y))
        
        manhatGhost = min(manhatGhost, 5)
        # print(0.5*manhatGhost)
        # print(1/(1+manhatFood))
        # print(0.5*successorGameState.getScore())

        score = 0.01*manhatGhost + 1/(0.001 + manhatFood) + (0.1 * successorGameState.getScore())
        # print("score is ", score)
        # print("------")

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

        # added field
        self.v = 0

class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):

        def value(gameState):

          numAgents = gameState.getNumAgents()

          self.index = (self.index + 1) % numAgents
          print("current agent is: ", self.index)
          print("current depth is: ", self.depth)
          if (self.depth <= 0 and self.index == 0) or (gameState.isWin() or gameState.isLose()):
            # assumes evalFn smart enough to return
            # positive inf for win, and negative inf for lose
            ret = self.evaluationFunction(gameState) 
            print("evaluated: ", ret)
            return ret 

          if self.index == 0:
            #next agent is pacman
            return maxValue(gameState)[0]
          else:
            return minValue(gameState)[0]


        def maxValue(gameState):
          self.depth -= 1
          v = float("-inf")
          bestAction = None

          legalActions = gameState.getLegalActions(self.index)
          print("LEGAL ACTIONS: ", legalActions)

          curr_idx = self.index
          curr_depth = self.depth

          for action in legalActions: # analogous to for each successor of state

            successorState = gameState.generateSuccessor(self.index, action)
            print("action in maxvalue is: ", action)
            successorValue = value(successorState)

            if (successorValue > v):
              v = successorValue
              bestAction = action

            self.index = curr_idx
            self.depth = curr_depth

            print("max value of going ", action, " is ", v)
            print("\n")


          return (v, bestAction)


        def minValue(gameState):
          v = float("+inf")
          bestAction = None

          legalActions = gameState.getLegalActions(self.index)
          print("LEGAL ACTIONS: ", legalActions)

          curr_idx = self.index
          curr_depth = self.depth
          for action in legalActions: # analogous to for each successor of state

            successorState = gameState.generateSuccessor(self.index, action)
            print("action in minvalue is: ", action)

            # numAgents = gameState.getNumAgents()
            successorValue = value(successorState)
            if (successorValue < v):
              v = successorValue
              bestAction = action

            self.index = curr_idx
            self.depth = curr_depth

            print("min value of going ", action, " is ", successorValue)
            print("\n")
          return (v, bestAction)


        print("----------------------------")

        curr_depth = self.depth
        v = maxValue(gameState)[1]
        self.index = 0
        self.depth = curr_depth

        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def value(gameState, alpha, beta):

          numAgents = gameState.getNumAgents()

          self.index = (self.index + 1) % numAgents
          # print("current agent is: ", self.index)
          # print("current depth is: ", self.depth)
          if (self.depth <= 0 and self.index == 0) or (gameState.isWin() or gameState.isLose()):
            ret = self.evaluationFunction(gameState) 
            # print("evaluated: ", ret)
            return ret 



          if self.index == 0:
            #next agent is pacman

            return maxValue(gameState, alpha, beta)[0]
          else:
            return minValue(gameState, alpha, beta)[0]


        def maxValue(gameState, alpha, beta):
          self.depth -= 1
          v = float("-inf")
          bestAction = None

          legalActions = gameState.getLegalActions(self.index)
          # print("LEGAL ACTIONS: ", legalActions)

          curr_idx = self.index
          curr_depth = self.depth

          for action in legalActions: # analogous to for each successor of state

            successorState = gameState.generateSuccessor(self.index, action)
            # print("action in maxvalue is: ", action)
            successorValue = value(successorState, alpha, beta)

            if (successorValue > v):
              v = successorValue
              bestAction = action

            if v > beta:
              self.index = curr_idx
              self.depth = curr_depth
              return (v, bestAction, alpha)

            alpha = max(alpha, v)

            self.index = curr_idx
            self.depth = curr_depth

            # print("max value of going ", action, " is ", v)
            # print("\n")


          return (v, bestAction, alpha)


        def minValue(gameState, alpha, beta):
          v = float("+inf")
          bestAction = None

          legalActions = gameState.getLegalActions(self.index)
          # print("LEGAL ACTIONS: ", legalActions)

          curr_idx = self.index
          curr_depth = self.depth
          for action in legalActions: # analogous to for each successor of state

            successorState = gameState.generateSuccessor(self.index, action)
            # print("action in minvalue is: ", action)

            # numAgents = gameState.getNumAgents()
            successorValue = value(successorState, alpha, beta)
            if (successorValue < v):
              v = successorValue
              bestAction = action

            if v < alpha:
              self.index = curr_idx
              self.depth = curr_depth
              return (v, bestAction, beta)

            beta = min(beta, v)

            self.index = curr_idx
            self.depth = curr_depth

            # print("min value of going ", action, " is ", successorValue)
            # print("\n")
          return (v, bestAction, beta)


        # print("----------------------------")

        alpha = float("-inf")
        beta = float("+inf")
        curr_depth = self.depth
        v = maxValue(gameState, alpha, beta)[1]
        self.index = 0
        self.depth = curr_depth

        return v




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction







"""
  Returns the minimax action from the current gameState using self.depth
  and self.evaluationFunction.

  Here are some method calls that might be useful when implementing minimax.

  gameState.getLegalActions(agentIndex):
    Returns a list of legal actions for an agent
    agentIndex=0 means Pacman, ghosts are >= 1

  gameState.generateSuccessor(agentIndex, action):
    Returns the successor game state after an agent takes an action

  gameState.getNumAgents():
    Returns the total number of agents in the game

  gameState.isWin():
    Returns whether or not the game state is a winning state

  gameState.isLose():
    Returns whether or not the game state is a losing state
"""

