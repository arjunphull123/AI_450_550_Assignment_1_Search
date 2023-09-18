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

    You do NOT need to change anything in this class, ever.
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
    return [s, s, w, s, w, w, s, w]




def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    "*** YOUR CODE HERE ***"
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    # Create infrastructure to search:
    class Node:
        def __init__(self, state, parent=None, action=None):
            self.state = state
            self.parent = parent
            self.action = action

        def get_path(self):
            """Gets the path from the initial state to the current state.

            :return: _description_
            """
            path = []  # initialize path list
            current_node = self  # set current node to expand
            while current_node is not None:  # will traverse the entire path
                if current_node.action:  # if there is an action to add
                    path.append(current_node.action)  # add the action to the path
                current_node = current_node.parent  # jump up one level to parent

            # at this point, path is a path of actions from the current state to
            # initial state. we need to reverse this

            path.reverse() # reverse the path
            return path

    # Use the newly-minted Node class!
    start = Node(problem.getStartState)  # start node is initalized with no parent or action

    # If the start node is a goal state, return an empty list
    if problem.isGoalState(start.state):
        return start.get_path()
    
    # So, the start node is not a goal. Bummer! Let's search:

    # Generic process is as follows: 
    # First, initialize the frontier as a stack (LIFO). 
    # Initialize an empty explored set.
    # Push the start node to the frontier. 
    
    frontier = util.Stack()  # initialize frontier
    explored = set()  # initialize explored set
    frontier.push(start)  # push the start node to the frontier
    
    # Then, do the following until the frontier is empty (or loop breaks with return):
    while not frontier.isEmpty():
        # Choose a leaf and remove (pop) it from the frontier.
        leaf = frontier.pop()
        # If the leaf contains a goal state then return the corresponding solution
        if problem.isGoalState(leaf.state):
            return leaf.get_path()
        # add the leaf (the node's state) to the explored set
        explored.add(leaf.state)

        # expand the chosen leaf, adding the resulting nodes to the frontier
        # only if not in the frontier or explored set
        # Expansion steps:
        # 1. Get successors
        successors = problem.getSuccessors(leaf.state)
        # 2. Make nodes for each successor
            # Note: successor is formatted as (state, action, stepCost)
        for successor in successors:
            state = successor[0]
            action = successor[1]
            succ_node = Node(state, action)
            # 3. If the node is not in the frontier or the explored set, add it to the frontier
            if succ_node.state not in explored and succ_node.state not in frontier.list:
                frontier.push(succ_node)
    # The gist of this is: the loop will repeat until a goal state is found or the frontier is empty.
    
    # If the code gets to this point, the frontier is empty
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    "*** YOUR CODE HERE ***"
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
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
