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

class Node:
    """This class defines the structure of a node for use in the following
    search functions. Each node has a state, a parent, an action, and a path
    cost.
    """

    def __init__(self, state, parent=None, action=None, path_cost=0, heuristic_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.heuristic_cost = heuristic_cost
        self.path_cost = path_cost
        if self.parent:
            self.path_cost += self.parent.path_cost

    def get_path(self):
        """Gets the path from the initial state to the current state.

        :return: A list of actions representing the path from the initial state
        to the current state.
        """
        path = []  # initialize path list
        current_node = self  # set current node to expand
        while current_node is not None:  # will traverse the entire path
            if current_node.action is not None:  # if there is an action to add
                path.append(current_node.action)  # add the action to the path
            current_node = current_node.parent  # jump up one level to parent
        path.reverse() # reverse the path
        return path
    

    def in_frontier(self, frontier):
        """Returns True if self.state is in frontier, False otherwise.

        :param frontier: A Stack, Queue, or PriorityQueue.
        :return: True if self.state is in frontier, False otherwise.
        """
        try:
            for node in frontier.list:  # iterate across the frontier list
                if node.state == self.state:  # compare states
                    return True
            return False
        except AttributeError:
            for (priority, count, item) in frontier.heap:  # iterate across the frontier heap
                if item.state == self.state:  # compare states
                    return True
            return False

def graphSearch(problem, frontier, heuristic=None):
    """This function conducts a graph search for the given problem. The strategy used
    is determined by the data structure of the given frontier, as follows:
    - If the frontier is a Stack, the function uses depth-first-search.
    - If the frontier is a Queue, the function uses breadth-first-search.
    - If the frontier is a PriorityQueue, the function uses uniform-cost-search.
    - If there is a heuristic function passed in, the function uses A* search.

    :param problem: A search problem.
    :param frontier: A Stack, Queue, or PriorityQueue.
    :return: A list of actions that lead to the problem's goal state.
    """
    start = Node(problem.getStartState())

    if problem.isGoalState(start.state):  # check for initial goal
        return start.get_path()
    
    if heuristic:
        start.heuristic_cost = heuristic(start.state, problem)  # get the heuristic cost of the node if using A*
    
    if isinstance(frontier, util.PriorityQueue):
        frontier.push(start, start.path_cost + start.heuristic_cost)  # push the start node with the path cost (+ heuristic cost if using A*)
    else:
        frontier.push(start)  # push the start node to the frontier
    explored = set()  # initialize explored set
    
    while not frontier.isEmpty():
        node = frontier.pop()  # pop a node

        if problem.isGoalState(node.state):  # goal-test
            return node.get_path()
        
        if node.state not in explored:  # expand the node if it has not been explored
            explored.add(node.state)  # add node to explored

            successors = problem.getSuccessors(node.state)  # expand node
            for successor in successors:
                state = successor[0]
                action = successor[1]
                cost = successor[2]
                child = Node(state, node, action, cost)  # create child node

                if child.state not in explored:
                    if isinstance(frontier, util.Stack):  # handle Stack
                        frontier.push(child)
                    elif isinstance(frontier, util.PriorityQueue):  # handle PriorityQueue
                        if heuristic:  # handle A*
                            child.heuristic_cost = heuristic(child.state, problem)
                        frontier.update(child, child.path_cost + child.heuristic_cost)  # A* is the same as uniform cost, except heuristic augments the cost
                    elif not child.in_frontier(frontier):  # handle Queue
                        frontier.push(child)

    # If no solution is returned:
    util.raiseNotDefined()

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
    return graphSearch(problem, util.Stack())


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    "*** YOUR CODE HERE ***"
    return graphSearch(problem, util.Queue())


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    "*** YOUR CODE HERE ***"
    return graphSearch(problem, util.PriorityQueue())


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    "*** YOUR CODE HERE ***"
    return graphSearch(problem, util.PriorityQueue(), heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
