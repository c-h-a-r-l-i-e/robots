from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np
import os
import re
import scipy.signal
import yaml
import time


# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)  # Any orientation is good.
START_POSE = np.array([-1.5, -1.5, 0.], dtype=np.float32)
MAX_ITERATIONS = 500

PLOT_SAMPLES = False


def sample_random_position(occupancy_grid):
  # Sample a valid random position (do not sample the yaw).
  # The corresponding cell must be free in the occupancy grid.
  position = np.random.random_sample(2)  * 4 - 2

  while not occupancy_grid.is_free(position):
    position = np.random.random_sample(2) * 4 - 2

  return position


def adjust_pose(node, final_position, occupancy_grid):
  final_pose = node.pose.copy()
  final_pose[:2] = final_position

  # Check whether there exists a simple path that links node.pose
  # to final_position. This function needs to return a new node that has
  # the same position as final_position and a valid yaw. The yaw is such that
  # there exists an arc of a circle that passes through node.pose and the
  # adjusted final pose. If no such arc exists (e.g., collision) return None.
  # Assume that the robot always goes forward.
  # Feel free to use the find_circle() function below.

  delta_position = final_position - node.pose[:2]

  # Path_yaw is the angle between the x axis and the line drawn from node to
  # final position
  path_yaw = np.arctan2(delta_position[Y], delta_position[X])

  delta_yaw = 2 * (node.pose[YAW] - path_yaw)
  final_pose[YAW] = node.pose[YAW] - delta_yaw
  final_node = Node(final_pose)

  centre, radius = find_circle(node, final_node)

  if PLOT_SAMPLES:
    occupancy_grid.draw()
    plt.plot(node.pose[X], node.pose[Y], 'o', color='red')
    plt.plot(final_pose[X], final_pose[Y], 'o', color='green')


  def get_arc_position(theta):
    position = np.zeros((2))
    position[X] = centre[X] + radius * np.cos(theta)
    position[Y] = centre[Y] + radius * np.sin(theta)
    return position

  start_theta = np.arctan2(node.pose[Y] - centre[Y], node.pose[X] - centre[X])
  clockwise = np.cross(node.direction, node.pose[[X,Y]] - centre) > 0
  end_theta = np.arctan2(final_pose[Y] - centre[Y], final_pose[X] - centre[X])
  if clockwise:
    end_theta = start_theta - np.mod(start_theta - end_theta, np.pi * 2)
  else:
    end_theta = start_theta + np.mod(end_theta - start_theta, np.pi * 2)

  # Calculate number of segments so we assume there's a robot every ROBOT_RADIUS/2 metres
  num_segments = int((2 * radius * np.pi * np.abs(end_theta - start_theta) / (np.pi * 2)) / (ROBOT_RADIUS / 2))
  
  midpoints = [get_arc_position(start_theta + (t * (end_theta - start_theta) / num_segments)) for t in range(num_segments)]

  valid = True
  for midpoint in midpoints:
    lower_left = occupancy_grid.get_index(
        np.array((midpoint[X] - ROBOT_RADIUS, midpoint[Y] - ROBOT_RADIUS)))
    upper_right = occupancy_grid.get_index(
        np.array((midpoint[X] + ROBOT_RADIUS, midpoint[Y] + ROBOT_RADIUS)))

    for i in range(lower_left[0], upper_right[0] + 1):
      for j in range(lower_left[1], upper_right[1] + 1):
        pos = occupancy_grid.get_position(i, j)

        if PLOT_SAMPLES:
          plt.plot(pos[X], pos[Y], 'b+')

        if not occupancy_grid.is_free(pos):
          valid = False
          break

      if not valid:
        break

    if not valid:
      break

  if PLOT_SAMPLES:
    plt.show()
  
  if not valid:
    return None
  else:
    return final_node

  

# Rewire as required for RRT*
def rewire(parent, child, occupancy_grid):
  # Find new orientation for child node, if not possible return None
  new_child = adjust_pose(parent, child.position, occupancy_grid)

  if new_child is None:
    return None

  new_child.cost = parent.cost + arc_distance(parent, child.position, occupancy_grid)
  
  # If an appropriate path exists, run algorithm on all of child's children
  for grandchild in child.neighbors:
    new_grandchild = rewire(new_child, grandchild, occupancy_grid)

    if new_grandchild is None:
      return None

    new_child.add_neighbor(new_grandchild)

  new_child.parent = parent

  # If all of those can be solved, commit the change, returning the new subtree, starting at child
  return new_child
  


# Defines an occupancy grid.
class OccupancyGrid(object):
  def __init__(self, values, origin, resolution):
    self._original_values = values.copy()
    self._values = values.copy()
    # Inflate obstacles (using a convolution).
    inflated_grid = np.zeros_like(values)
    inflated_grid[values == OCCUPIED] = 1.
    w = 2 * int(ROBOT_RADIUS / resolution) + 1
    inflated_grid = scipy.signal.convolve2d(inflated_grid, np.ones((w, w)), mode='same')
    self._values[inflated_grid > 0.] = OCCUPIED
    self._origin = np.array(origin[:2], dtype=np.float32)
    self._origin -= resolution / 2.
    assert origin[YAW] == 0.
    self._resolution = resolution

  @property
  def values(self):
    return self._values

  @property
  def resolution(self):
    return self._resolution

  @property
  def origin(self):
    return self._origin

  def draw(self):
    plt.imshow(self._original_values.T, interpolation='none', origin='lower',
               extent=[self._origin[X],
                       self._origin[X] + self._values.shape[0] * self._resolution,
                       self._origin[Y],
                       self._origin[Y] + self._values.shape[1] * self._resolution])
    plt.set_cmap('gray_r')

  def get_index(self, position):
    idx = ((position - self._origin) / self._resolution).astype(np.int32)
    if len(idx.shape) == 2:
      idx[:, 0] = np.clip(idx[:, 0], 0, self._values.shape[0] - 1)
      idx[:, 1] = np.clip(idx[:, 1], 0, self._values.shape[1] - 1)
      return (idx[:, 0], idx[:, 1])
    idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
    idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
    return tuple(idx)

  def get_position(self, i, j):
    return np.array([i, j], dtype=np.float32) * self._resolution + self._origin

  def is_occupied(self, position):
    return self._values[self.get_index(position)] == OCCUPIED

  def is_free(self, position):
    return self._values[self.get_index(position)] == FREE


# Defines a node of the graph.
class Node(object):
  def __init__(self, pose):
    self._pose = pose.copy()
    self._neighbors = []
    self._parent = None
    self._cost = 0.

  @property
  def pose(self):
    return self._pose

  def add_neighbor(self, node):
    self._neighbors.append(node)

  def remove_neighbor(self, node):
    if node in self._neighbors:
      self._neighbors.remove(node)

  @property
  def parent(self):
    return self._parent

  @parent.setter
  def parent(self, node):
    self._parent = node

  @property
  def neighbors(self):
    return self._neighbors


  @property
  def position(self):
    return self._pose[:2]

  @property
  def yaw(self):
    return self._pose[YAW]
  
  @property
  def direction(self):
    return np.array([np.cos(self._pose[YAW]), np.sin(self._pose[YAW])], dtype=np.float32)

  @property
  def cost(self):
      return self._cost

  @cost.setter
  def cost(self, c):
    self._cost = c


def arc_distance(start_node, final_position, occupancy_grid):
  final_pose = start_node.pose.copy()
  final_pose[:2] = final_position

  delta_position = final_position - start_node.position
  path_yaw = np.arctan2(delta_position[Y], delta_position[X])
  delta_yaw = 2 * (start_node.pose[YAW] - path_yaw)
  final_pose[YAW] = start_node.pose[YAW] - delta_yaw
  final_node = Node(final_pose)
  centre, radius = find_circle(start_node, final_node)

  start_theta = np.arctan2(start_node.position[Y] - centre[Y], start_node.position[X] - centre[X])
  clockwise = np.cross(start_node.direction, start_node.position - centre) > 0
  end_theta = np.arctan2(final_position[Y] - centre[Y], final_position[X] - centre[X])
  if clockwise:
    end_theta = start_theta - np.mod(start_theta - end_theta, np.pi * 2)
  else:
    end_theta = start_theta + np.mod(end_theta - start_theta, np.pi * 2)

  distance = 2 * radius * np.pi * np.abs(end_theta - start_theta) / (np.pi * 2)
  return distance


def rrt_star(start_pose, goal_position, occupancy_grid):
  # RRT* builds a graph one node at a time.
  graph = []
  start_node = Node(start_pose)
  final_node = None
  if not occupancy_grid.is_free(goal_position):
    print('Goal position is not in the free space.')
    return start_node, final_node
  graph.append(start_node)
  for _ in range(MAX_ITERATIONS): 
    position = sample_random_position(occupancy_grid)
    # With a random chance, draw the goal position.
    if np.random.rand() < .05:
      position = goal_position

    # Find closest node in graph.
    # In practice, one uses an efficient spatial structure (e.g., quadtree).
    potential_parent = sorted(((n, np.linalg.norm(position - n.position)) for n in graph), key=lambda x: x[1])

    # -----------------------------------------
    # | Find lowest cost parent               |
    # -----------------------------------------

    # Pick a node at least some distance away but not too far.
    # We also verify that the angles are aligned (within pi / 4).
    u = None
    lowest_cost = np.inf
    for n, d in potential_parent:
      if d > .2 and d < 1.5 and n.direction.dot(position - n.position) / d > 0.70710678118:
        # find neighbor which gives lowest cost
        actual_d = arc_distance(n, position, occupancy_grid)
        if n.cost + actual_d < lowest_cost:
          u = n
          lowest_cost = n.cost + actual_d

    if u is None:
      continue

    v = adjust_pose(u, position, occupancy_grid)
    if v is None:
      continue

    v.cost = lowest_cost
    u.add_neighbor(v)
    v.parent = u
    graph.append(v)

    # -----------------------------------------
    # | Rewire                                |
    # -----------------------------------------
    def remove_from_graph(node):
      for neighbor in node.neighbors:
        remove_from_graph(neighbor)
      if node in graph:
        graph.remove(node)

    def add_to_graph(node):
      for neighbor in node.neighbors:
        add_to_graph(neighbor)
      if node not in graph:
        graph.append(node)


    # For each node not too far away.
    # We also verify that the angles are aligned (within pi / 4).
    for n, d in potential_parent:
      if d > 0.2 and d < 1.5 and v.direction.dot(n.position - v.position) / d > 0.70710678118:
        actual_d = arc_distance(n, position, occupancy_grid)
        if v.cost + actual_d < n.cost:
          new_n = rewire(v, n, occupancy_grid)
          if new_n is not None:
            remove_from_graph(n)
            n.parent.remove_neighbor(n)
            add_to_graph(new_n)
            v.add_neighbor(new_n)

    if np.linalg.norm(v.position - goal_position) < .2 and (final_node is None or v.cost < final_node.cost):
      final_node = v
  return start_node, final_node


def find_circle(node_a, node_b):
  def perpendicular(v):
    w = np.empty_like(v)
    w[X] = -v[Y]
    w[Y] = v[X]
    return w
  db = perpendicular(node_b.direction)
  dp = node_a.position - node_b.position
  t = np.dot(node_a.direction, db)
  if np.abs(t) < 1e-3:
    # By construction node_a and node_b should be far enough apart,
    # so they must be on opposite end of the circle.
    center = (node_b.position + node_a.position) / 2.
    radius = np.linalg.norm(center - node_b.position)
  else:
    radius = np.dot(node_a.direction, dp) / t
    center = radius * db + node_b.position
  return center, np.abs(radius)


def read_pgm(filename, byteorder='>'):
  """Read PGM file."""
  with open(filename, 'rb') as fp:
    buf = fp.read()
  try:
    header, width, height, maxval = re.search(
        b'(^P5\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n]\s)*)', buf).groups()
  except AttributeError:
    raise ValueError('Invalid PGM file: "{}"'.format(filename))
  maxval = int(maxval)
  height = int(height)
  width = int(width)
  img = np.frombuffer(buf,
                      dtype='u1' if maxval < 256 else byteorder + 'u2',
                      count=width * height,
                      offset=len(header)).reshape((height, width))
  return img.astype(np.float32) / 255.


def draw_solution(start_node, final_node=None):
  ax = plt.gca()

  def draw_path(u, v, arrow_length=.1, color=(.8, .8, .8), lw=1):
    du = u.direction
    plt.arrow(u.pose[X], u.pose[Y], du[0] * arrow_length, du[1] * arrow_length,
              head_width=.05, head_length=.1, fc=color, ec=color)
    dv = v.direction
    plt.arrow(v.pose[X], v.pose[Y], dv[0] * arrow_length, dv[1] * arrow_length,
              head_width=.05, head_length=.1, fc=color, ec=color)
    center, radius = find_circle(u, v)
    du = u.position - center
    theta1 = np.arctan2(du[1], du[0])
    dv = v.position - center
    theta2 = np.arctan2(dv[1], dv[0])
    # Check if the arc goes clockwise.
    if np.cross(u.direction, du).item() > 0.:
      theta1, theta2 = theta2, theta1
    ax.add_patch(patches.Arc(center, radius * 2., radius * 2.,
                             theta1=theta1 / np.pi * 180., theta2=theta2 / np.pi * 180.,
                             color=color, lw=lw))

  points = []
  s = [(start_node, None)]  # (node, parent).
  while s:
    v, u = s.pop()
    if hasattr(v, 'visited'):
      continue
    v.visited = True
    # Draw path from u to v.
    if u is not None:
      draw_path(u, v)
    points.append(v.pose[:2])
    for w in v.neighbors:
      s.append((w, v))

  points = np.array(points)
  plt.scatter(points[:, 0], points[:, 1], s=10, marker='o', color=(.8, .8, .8))
  if final_node is not None:
    plt.scatter(final_node.position[0], final_node.position[1], s=10, marker='o', color='k')
    # Draw final path.
    v = final_node
    while v.parent is not None:
      draw_path(v.parent, v, color='k', lw=2)
      v = v.parent


if __name__ == '__main__':

 
  parser = argparse.ArgumentParser(description='Uses RRT to reach the goal.')
  parser.add_argument('--map', action='store', default='map', help='Which map to use.')
  args, unknown = parser.parse_known_args()

  # Load map.
  with open(args.map + '.yaml') as fp:
    data = yaml.load(fp)
  img = read_pgm(os.path.join(os.path.dirname(args.map), data['image']))
  occupancy_grid = np.empty_like(img, dtype=np.int8)
  occupancy_grid[:] = UNKNOWN
  occupancy_grid[img < .1] = OCCUPIED
  occupancy_grid[img > .9] = FREE
  # Transpose (undo ROS processing).
  occupancy_grid = occupancy_grid.T
  # Invert Y-axis.
  occupancy_grid = occupancy_grid[:, ::-1]
  occupancy_grid = OccupancyGrid(occupancy_grid, data['origin'], data['resolution'])

  # Run RRT.
  total_cost = 0
  total_time = 0
  for i in range(1):
    t = time.clock()
    start_node, final_node = rrt_star(START_POSE, GOAL_POSITION, occupancy_grid)
    total_time += time.clock() - t
    total_cost += final_node.cost

  print("Average time: {}".format(total_time/40))
  print("Average cost: {}".format(total_cost/40))
    

  # Plot environment.
  fig, ax = plt.subplots()
  occupancy_grid.draw()
  plt.scatter(.3, .2, s=10, marker='o', color='green', zorder=1000)
  draw_solution(start_node, final_node)
  plt.scatter(START_POSE[0], START_POSE[1], s=10, marker='o', color='green', zorder=1000)
  plt.scatter(GOAL_POSITION[0], GOAL_POSITION[1], s=10, marker='o', color='red', zorder=1000)
  
  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-.5 - 2., 2. + .5])
  plt.ylim([-.5 - 2., 2. + .5])
  plt.show()
  
