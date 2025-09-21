import math

# Function to calculate Euclidean distance between two nodes
def distance(node1, node2):
    x1, y1 = node1
    x2, y2 = node2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Example coordinates for 19 nodes
nodes = {
    '0': (35, 540),
    '1': (59, 497),
    '2': (55, 406),
    '3': (60, 296),
    '4': (63, 225),
    '5': (79, 157),
    '6': (123, 155),
    '7': (374, 174),
    '8': (422, 252),
    '9': (420, 320),
    '10': (419, 403),
    '11': (341, 399),
    '12': (243, 399),
    '13': (240, 490),
    '14': (137, 494),
    '15': (134, 299),
    '16': (243, 305),
    '17': (247, 241),
    '18': (341, 312),
}

# Example: Calculate distance between node '0' and node '1'
dist = distance(nodes['16'], nodes['18'])
print(f"Distance between node 0 and node 1: {dist:.2f}")

# Loop through all pairs if you want distances between all nodes
# for i in range(len(nodes)):
#     for j in range(i + 1, len(nodes)):
#         nodeA, nodeB = f'{i}', f'{j}'
#         dist = distance(nodes[nodeA], nodes[nodeB])
#         print(f"Distance between node {nodeA} and node {nodeB}: {dist:.2f}")
