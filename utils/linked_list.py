class Node(object):
    def __init__(self, val, next_node, prev_node):
        self.val = val
        self.next = next_node
        self.prev = prev_node


class LinkedListMetric(object):
    def __init__(self, max_length=1000):
        self.head = None
        self.tail = None
        self.total_length = 0
        self.total_val = 0
        self.avg = 0
        self.max_length = max_length
        self.val = None

    def put_(self, val):
        current = Node(val, None, self.tail)
        if self.head is None:
            self.head = current
        if self.tail is not None:
            self.tail.next = current
        self.tail = current
        self.total_val += val
        if self.total_length >= self.max_length:
            self.total_val -= self.head.val
            self.head = self.head.next
            self.head.prev = None
        else:
            self.total_length += 1
        self.avg = self.total_val / self.total_length
        self.val = val
