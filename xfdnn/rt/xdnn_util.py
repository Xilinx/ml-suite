#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2019, Xilinx, Inc.
#

from collections import defaultdict

######################################################
## utility functions for union-find structure
######################################################
class UnionFind(object):
  def __init__(self, size):
    self.array = [i for i in range(size)]
    self.weight = [1 for i in range(size)]

  def root(self, i):
    while (self.array[i] != i):
      i = self.array[i]
    return i

  def find(self, i, j):
    return self.root(i) == self.root(j)

  def union(self, i, j):
    root_i = self.root(i)
    root_j = self.root(j)
    if root_i == root_j:
      return
    if self.weight[root_i] < self.weight[root_j]:
      self.array[root_i] = self.array[root_j]
      self.weight[root_j] += self.weight[root_i]
    else:
      self.array[root_j] = self.array[root_i]
      self.weight[root_i] += self.weight[root_j]

  def components(self):
    comp = defaultdict(list)
    for i, parent in enumerate(self.array):
      root_i = self.root(parent)
      if i != root_i:
        comp[root_i].append(i)
    return comp



######################################################
## utility functions to find Lowest Common Scope of nodes
######################################################
class TrieNode(object):
  def __init__(self, key=None):
    self.key = key
    self.children = []

    # isEndOfList is True if node represent the end of the list
    self.isEndOfList = False

  def __contains__(self, child_key):
    return any([child_key==child.key for child in self.children])

  def __getitem__(self, child_key):
    if child_key not in self:
      return None
    return [child for child in self.children if child_key == child.key][0]

  def add(self, child_key):
    self.children.append(TrieNode(child_key))


class Trie(object):
  def __init__(self, name_list=[], name_sep='/'):
    self.name_sep = name_sep
    self.root = self.newNode()
    if name_list:
      for name in name_list:
        self.insert(name)

  def newNode(self):
    # Returns new trie node (initialized to NULLs)
    return TrieNode()

  def insert(self, name):
    # If not present, inserts name scopes into trie
    # If the scope is prefix of trie node,
    # just marks leaf node
    root = self.root
    for scope in name.split(self.name_sep):
      # if current character is not present
      if scope not in root:
        root.add(scope)
      root = root[scope]

    # mark last node as leaf
    root.isEndOfList = True

  def search(self, name):
    # Search name in the trie
    # Returns true if name presents
    # in trie, else false
    root = self.root
    for scope in name.split(self.name_sep):
      if scope not in root:
        return False
      root = root[scope]

    return (root != None and root.isEndOfList)

  def lcs(self):
    ## find lowest common scope
    root = self.root
    ret = []
    while len(root.children) == 1:
      ret.append(root.key)
      root = root.children[0]
    ret.append(root.key)
    return self.name_sep.join(ret[1:])


######################################################
## utility functions to convert between dict and attr
######################################################
class dict2attr(dict):
  DEFAULT = None

  def __init__(self, mapping=None):
    if mapping is None:
      return
    elif isinstance(mapping, dict):
      items = mapping.items
    elif hasattr(mapping, '__dict__'):
      items = mapping.__dict__.items
    else:
      raise TypeError('expected dict')
    for key, value in items():
      self[key] = value

  def __setitem__(self, key, value):
    if isinstance(value, dict) and not isinstance(value, dict2attr):
      value = dict2attr(value)
    super(dict2attr, self).__setitem__(key, value)

  def __getitem__(self, key):
    #found = self.get(key, dict2attr.DEFAULT)
    found = super(dict2attr, self).get(key, dict2attr.DEFAULT)
    # if found is dict2attr.DEFAULT:
    #   found = dict2attr()
    #   super(dict2attr, self).__setitem__(key, found)
    return found

  def get(self, key, default=None):
    val = self[key]
    return val if val is not dict2attr.DEFAULT else default

  def update(self, other):
    if hasattr(other, '__dict__'):
      other = other.__dict__
    if not isinstance(other, dict):
      raise TypeError('expected dict')
    super(dict2attr, self).update(other)

  __setattr__, __getattr__ = __setitem__, __getitem__
