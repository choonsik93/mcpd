import numpy as np

class Node:
    def __init__(self, R, v, parent=None):
        self.v = v
        self.R = R
        self.accjoint = np.copy(self.v)
        self.accR = np.copy(self.R)
        self.parent = parent
        self.children = []
        if parent is None:
            self.d = np.zeros(3)
        if parent is not None:
            self.parent.children.append(self)
        
def skim_node(node, data_type='y'):
    if len(node.children) == 0:
        if data_type == 'y':
            return node.y
        elif data_type == 'ty':
            return node.ty
        elif data_type == 'index':
            return node.index
        elif data_type == 'accjoint':
            return [node.accjoint]
        elif data_type == 'v':
            return [node.v]
    else:
        if data_type == 'y':
            value = node.y
            for child in node.children:
                value = np.concatenate((value, skim_node(child, data_type='y')))
            return value
        elif data_type == 'ty':
            value = node.ty
            for child in node.children:
                value = np.concatenate((value, skim_node(child, data_type='ty')))
            return value
        elif data_type == 'index':
            value = node.index
            for child in node.children:
                value = np.concatenate((value, skim_node(child, data_type='index')))
            return value
        elif data_type == 'accjoint':
            value = [node.accjoint]
            for child in node.children:
                value = np.concatenate((value, skim_node(child, data_type='accjoint')))
                #value = value + skim_node(child, data_type='accjoint')
            return value
        elif data_type == 'v':
            value = [node.v]
            for child in node.children:
                value = np.concatenate((value, skim_node(child, data_type='v')))
                #value = value + skim_node(child, data_type='accjoint')
            return value
    
def update_ty(node):
    if node.parent is None:
        node.accR = np.identity(3)
        node.accjoint = node.d + node.v
    else:
        parent = node.parent
        node.accR = np.dot(parent.accR, node.R)
        node.accjoint = parent.accjoint + np.dot(parent.accR, (node.v - parent.v))
        
    if len(node.children) == 0:
        return
    else:
        for child in node.children:
            update_ty(child)

class Tree:
    def __init__(self, root):
        self.root = root