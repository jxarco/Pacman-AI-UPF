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
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        successorGameState = currentGameState.generatePacmanSuccessor(action) # SUCCESSOR STATE
        newPos = successorGameState.getPacmanPosition() # POS X,Y
        newFood = successorGameState.getFood() # FOOD GRID (TRUE/FALSE)
        newGhostStates = successorGameState.getGhostStates() # WHERE ARE GHOSTS
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        score =     -1000000
        dist =       1000000

        """ Taking foods as lists """
        newFoodList = newFood.asList()
        oldFoodList = currentGameState.getFood().asList()

        # If PACMAN stops:
        if action == 'Stop':
          return score

        # If new state is food:
        if len(newFoodList) < len(oldFoodList):
          score += 500

        # If PACMAN find the ghost
        ghostDistance = manhattanDistance(successorGameState.getGhostPosition(1), newPos)
        score += max(2, ghostDistance)

        dist = 1000000
        for food in newFoodList:
          dist = min(dist, manhattanDistance(food, newPos))
          
        score -= dist * 10 

        if successorGameState.isWin():
          score += 1000000

        return score
        return successorGameState.getScore()

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

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"

        """ Number of ghosts in game """
        NG = gameState.getNumAgents() - 1

        def maxAgent(gameState, depth):

          """ If game is finished """
          if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

          # Initialize best action and score
          # v = -INF in max
          bestAction = None
          bestScore = float("-inf")

          legalActions = gameState.getLegalActions(0) # 0 is the index for pacman
          
          """ For each action we have to obtain max score of min movements """
          for action in legalActions:
            successorGameState = gameState.generateSuccessor(0, action)
            v = minAgent(successorGameState, depth, 1)
            # Update best max score
            if(v > bestScore):
              bestScore = v
              bestAction = action

          # Recursive calls have finished -> depth = initial depth -> return best action
          if depth == 0:
            return bestAction
          # We are in different depth, we need to return a score
          else:
            return bestScore

        def minAgent(gameState, depth, ghost):

          if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

          # Initialize score
          # v = INF in min
          bestScore = float("inf")
          # Legal actions for selected ghost
          legalActions = gameState.getLegalActions(ghost)
          
          for action in legalActions:
            successorGameState = gameState.generateSuccessor(ghost, action)
            if(ghost < NG):
              # There are still ghosts to move
              # Using ghost + 1 to select the next ghost
              v = minAgent(successorGameState, depth, ghost + 1) # returns a score
            else:
              # Last ghost -> next turn is for pacman
              if(depth == self.depth - 1): # IF IT IS A TERMINAL
                v = self.evaluationFunction(successorGameState)
              else:
                # If it is not a terminal
                v = maxAgent(successorGameState, depth + 1) # returns a score
            
            # Update best min score
            bestScore = min(v, bestScore)

          return bestScore


        # RETURN AN ACTION
        return maxAgent(gameState, 0) # depth = 0

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        """ Number of ghosts in game """
        NG = gameState.getNumAgents() - 1
        alpha = float("-inf")
        beta = float("inf")

        def maxAgent(gameState, depth, alpha, beta):

          """ If game is finished """
          if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

          # Initialize best action and score
          # v = -INF in max
          bestAction = None
          bestScore = float("-inf")

          legalActions = gameState.getLegalActions(0) # 0 is the index for pacman
          
          """ For each action we have to obtain max score of min movements """
          for action in legalActions:
            successorGameState = gameState.generateSuccessor(0, action)
            v = minAgent(successorGameState, depth, 1, alpha, beta)
            # Update best max score
            if(v > bestScore):
              bestScore = v
              bestAction = action

            if(bestScore > beta):
              return bestScore
            alpha = max(alpha, bestScore)

          # Recursive calls have finished -> depth = initial depth -> return best action
          if depth == 0:
            return bestAction
          # We are in different depth, we need to return a score
          else:
            return bestScore

        def minAgent(gameState, depth, ghost, alpha, beta):

          if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

          # Initialize score
          # v = INF in min
          bestScore = float("inf")
          # Legal actions for selected ghost
          legalActions = gameState.getLegalActions(ghost)
          
          for action in legalActions:
            successorGameState = gameState.generateSuccessor(ghost, action)
            if(ghost < NG):
              # There are still ghosts to move
              # Using ghost + 1 to select the next ghost
              v = minAgent(successorGameState, depth, ghost + 1, alpha, beta) # returns a score
            else:
              # Last ghost -> next turn is for pacman
              if(depth == self.depth - 1): # IF IT IS A TERMINAL
                v = self.evaluationFunction(successorGameState)
              else:
                # If it is not a terminal
                v = maxAgent(successorGameState, depth + 1, alpha, beta) # returns a score
            
            # Update best min score
            bestScore = min(v, bestScore)
            
            if(bestScore < alpha):
              return bestScore
            beta = min(beta, bestScore)

          return bestScore


        # RETURN AN ACTION
        return maxAgent(gameState, 0, alpha, beta) # depth = 0

        util.raiseNotDefined()

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

