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
    """This class defines the structure of a node for use in the graph search function.
        Each node has a state (its position), a parent (the node before it), an action
        (the action required to get there from its parent), and a path cost (the cost 
        to get to that node).
    """

    def __init__(self, state, parent=None, action=None, path_cost=0, heuristic_cost=0):
        self.state = state
        self.parent = parent  # defaults to None for starting node
        self.action = action  # defaults to None
        self.heuristic_cost = heuristic_cost  # h(n)
        self.path_cost = path_cost  # g(n)
        # note: f(n) = g(n) + h(n)
        if self.parent:  # because self.path_cost passed in is just the incremental cost,
            self.path_cost += self.parent.path_cost  # need to account for parent's cost

    def in_frontier(self, frontier):
        """When using breadth-first search, checks if a given state is already in
        the frontier Queue.

        :param frontier: A Queue.
        :return: True if self.state is in frontier, False otherwise.
        """
        for node in frontier.list:  # iterate across the frontier list
            if node.state == self.state:  # compare states
                return True
        return False  # returns if there are no matches

    def get_path(self):
        """Gets the path from the starting state to the current state.

        :return: A list of actions representing the path from the starting state
        to the current state.
        """
        path = util.Queue()  # initialize FIFO path
        current_node = self  # intialize the current node

        while current_node:  # recursively adds the parent's action to path
            # this loop will break once it reaches the starting node of the problem
            # (when the current node has no parent)
            if current_node.action:  # if there is an action to add
                path.push(current_node.action)  # add the action to the path
            current_node = current_node.parent  # jump up one level to parent
        return path.list  # return the list of the queue (in order from last added to first added)

def graphSearch(problem, frontier, heuristic=None):
    """This function conducts a generic graph search for the given problem. The strategy used
    is determined by the data structure of the given frontier, as follows:
    - If the frontier is a Stack, the function uses depth-first-search.
    - If the frontier is a Queue, the function uses breadth-first-search.
    - If the frontier is a PriorityQueue with no heuristic passed in, the function
      uses uniform-cost-search.
    - If there is a heuristic function passed in, the function uses A* search.

    :param problem: A search problem.
    :param frontier: A Stack, Queue, or PriorityQueue.
    :return: A list of actions that lead to the problem's goal state.
    """
    start = Node(problem.getStartState())  # get the start state

    if heuristic:  # if there is a heuristic passed in, use A* search
        start.heuristic_cost = heuristic(start.state, problem)  # get the heuristic cost of the node (to augment the path cost)
    
    # initialize the frontier using the initial state of problem
    if isinstance(frontier, util.PriorityQueue):
        frontier.push(start, start.path_cost + start.heuristic_cost)  # push the start node with the path cost (+ heuristic cost if using A*)
    else:
        frontier.push(start)  # push the start node to the frontier
    
    explored = set()  # initialize the explored set to be empty
    
    while not frontier.isEmpty():  # loop do; if the frontier is empty then return failure
        node = frontier.pop()  # choose a leaf node and remove it from the frontier

        if problem.isGoalState(node.state):  # if the node contains a goal state
            return node.get_path()  # then return the corresponding solution
        
        if node.state not in explored:  # expand the node if it has not been explored
            explored.add(node.state)  # add the node to the explored set

            # expand the chosen node:
            successors = problem.getSuccessors(node.state)  # get successor states from the node
            
            for successor in successors:  # iterate through the successors
                state = successor[0]  # unpack the state
                action = successor[1]  # unpack the action
                cost = successor[2]  # unpack the cost (defaults to 0 for DFS and BFS)
                child = Node(state, node, action, cost)  # create child node

                # add the resulting nodes to the frontier only if not in the frontier or explored set
                if child.state not in explored:  # only proceed if node is not in explored set
                    if isinstance(frontier, util.Stack):
                        # using DFS; the child is not in the frontier, so no need to check
                        frontier.push(child)
                    elif isinstance(frontier, util.PriorityQueue):
                        # using UCS or A*
                        if heuristic:  # using A*
                            child.heuristic_cost = heuristic(child.state, problem)  # get the heuristic cost
                        # otherwise, heuristic_cost defaults to 0
                        frontier.update(child, child.path_cost + child.heuristic_cost)  # only pushes the state if not already in frontier; see util.py
                    elif not child.in_frontier(frontier):  # using BFS; must check if state is in the frontier
                        frontier.push(child)  # adds child to frontier only if not already in frontier

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
    # conduct graph search with a Stack (LIFO)
    return graphSearch(problem, util.Stack())


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    "*** YOUR CODE HERE ***"
    # conduct graph search with a Queue (FIFO)
    return graphSearch(problem, util.Queue())


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    "*** YOUR CODE HERE ***"
    # conduct graph search with a priority queue and no heuristic
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
    # identical to UCS, except with a heuristic
    return graphSearch(problem, util.PriorityQueue(), heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
