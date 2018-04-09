import itertools, functools

def partition(pred, iterable):
    'Use a predicate to partition entries into false entries and true entries'
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    # Direct from the recipes in itertools documentation
    t1, t2 = itertools.tee(iterable)
    return itertools.filterfalse(pred, t1), filter(pred, t2)

def prob_tree_with_check(data,ini,visited=frozenset()) -> (float, tuple):
    """Generator of all end points of the probability tree contained 
       in data, starting with ini. Check if a previously visited branch
       of the tree is visited again and raise RuntimeError in that case"""
    if ini in visited:
        raise RuntimeError("Branch allready visited: %r"%ini)
    visited = visited.union((ini,))
    for prob,path in data[ini]:
        no_more,more = map(tuple,partition(lambda x: x in data,path))
        if more:
            for node in itertools.product( *[prob_tree_with_check(data,x,visited) for x in more] ):
                new_prob,new_path = functools.reduce(lambda acum,new: (acum[0]*new[0],acum[1]+new[1]),node,(prob,tuple()))
                yield new_prob, no_more + new_path
        else:
            yield prob, no_more

mydata_bad = {1: [[.9, [2,3]], [.1, [4,5]]],
          4: [[.2, [6,7]], [.5, [8,9]], [.3, [10,11,12]]],
          5: [[.4, [13,14]], [.6, [15,16,1]]] # <-- try to go back to 1
          }
