# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:""

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    """ DFS -> Stack  """
    stack = util.Stack()
    close = set()
    """ pathToGoal will store path data """
    pathToGoal = []
    startState = problem.getStartState()
    """
    Creating a node from a state (With no action, no cost, and no parent)
    """
    startNode = startState, '', 0, None
    stack.push( startNode )

    while not stack.isEmpty():
        # Take out last element inserted in the stack
        current = stack.pop()
        if problem.isGoalState( current[0] ):
            """ Backtracking to get the full path """
            while current[3] != None:
                # Add action
                pathToGoal.append( current[1] )
                # Update node to get next parent
                current = current[3]
                """ [::-1 to get the reverse path] """
            return pathToGoal[::-1]
        else:
            if current[0] not in close:
                close.add( current[0] )
                # Get successors of current state
                successors = problem.getSuccessors( current[0] )
                for s in successors:
                    node = s
                    # Adding parent to the tuple
                    node = node + (current, )
                    # Add successors to the stack
                    stack.push(node)
            else:
                # If not, we do not have to expand again
                continue
        
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    """ BFS -> Queue  """
    queue = util.Queue()
    """ If we use a set() we can not use NOT IN in corners problem """
    close = []
    """ pathToGoal will store path data """
    pathToGoal = []
    startState = problem.getStartState()
    """
    Creating a node from a state (With no action, no cost, and no parent)
    """
    startNode = startState, '', 0, None
    queue.push( startNode )

    while not queue.isEmpty():
        # Take out first element in the queue
        current = queue.pop()
        if problem.isGoalState( current[0] ):
            """ Backtracking to get the full path """
            while current[3] != None:
                # Add action
                pathToGoal.append( current[1] )
                # Update node to get next parent
                current = current[3]
                """ [::-1 to get the reverse path] """
            return pathToGoal[::-1]
        else:
            if current[0] not in close:
                close.append( current[0] )
                # Get successors of current state
                successors = problem.getSuccessors( current[0] )
                for s in successors:
                    node = s
                    # Adding parent to the tuple
                    node = node + (current, )
                    # Add successors to the queue
                    queue.push(node)
            else:
                # If not, we do not have to expand again
                continue
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    p_queue = util.PriorityQueue()
    close = set()
    """ pathToGoal will store path data """
    pathToGoal = []
    startState = problem.getStartState()
    """
    Creating a node from a state (With no action, no cost, and no parent)
    """
    startNode = startState, '', 0, None
    p_queue.push( startNode, 0)

    while not p_queue.isEmpty():
        current = p_queue.pop()
        if problem.isGoalState( current[0] ):
            """ Backtracking to get the full path """
            while current[3] != None:
                # Add action
                pathToGoal.append( current[1] )
                # Update node to get next parent
                current = current[3]
                """ [::-1 to get the reverse path] """
            return pathToGoal[::-1]
        else:
            if current[0] not in close:
                close.add( current[0] )
                # Get successors of current state
                successors = problem.getSuccessors( current[0] )
                for s in successors:
                    node = s
                    """ Adding parent to the tuple """
                    node = node + (current, )
                    """ Using aux_node to do the backtracking """
                    aux_node = node
                    count = aux_node[2]
                    """
                    The next while will calculate f(n) = priority in the QUEUE
                    doing the backtracking
                    """
                    while aux_node[3] != None:
                        aux_node = aux_node[3]
                        count += aux_node[2]
                    # Add node to the priority queue
                    p_queue.push(node, count)
            else:
                # If not, we do not have to expand again
                continue
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    p_queue = util.PriorityQueue()
    close = []
    """ pathToGoal will store path data """
    pathToGoal = []
    startState = problem.getStartState()
    """
    Creating a node from a state (With no action, no cost, and no parent)
    """
    startNode = startState, '', 0, None
    p_queue.push( startNode, 0)

    while not p_queue.isEmpty():
        current = p_queue.pop()
        if problem.isGoalState( current[0] ):
            """ Backtracking to get the full path """
            while current[3] != None:
                # Add action
                pathToGoal.append( current[1] )
                # Update node to get next parent
                current = current[3]
                """ [::-1 to get the reverse path] """
            return pathToGoal[::-1]
        else:
            if current[0] not in close:
                close.append( current[0] )
                # Get successors of current state
                successors = problem.getSuccessors( current[0] )
                for s in successors:
                    node = s
                    """ Adding parent to the tuple """
                    node = node + (current, )
                    """ Using aux_node to do the backtracking """
                    aux_node = node
                    count = aux_node[2]
                    """
                    The next while will calculate f(n) = priority in the QUEUE
                    doing the backtracking
                    """
                    while aux_node[3] != None:
                        aux_node = aux_node[3]
                        count += aux_node[2]
                    """ Only difference is the sum of the heuristic """
                    p_queue.push(node, count + heuristic(node[0], problem))
            else:
                # If not, we do not have to expand again
                continue
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
