"""Implementations of all on-path strategies"""
from __future__ import division
import random

import networkx as nx

from icarus.registry import register_strategy
from icarus.util import inheritdoc, path_links

from .base import Strategy
import example

from bitarray import bitarray
import mmh3
import time



class BloomFilter(set):

    def __init__(self, size, hash_count):
        super(BloomFilter, self).__init__()
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)
        self.size = size
        self.hash_count = hash_count

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.bit_array)

    def add(self, item):
        for ii in range(self.hash_count):
            index = mmh3.hash(item, ii) % self.size
            self.bit_array[index] = 1

        return self

    def __contains__(self, item):
        out = True
        for ii in range(self.hash_count):
            index = mmh3.hash(item, ii) % self.size
            if self.bit_array[index] == 0:
                out = False

        return out


__all__ = [
       'Partition',
       'Edge',
       'Non_cache',
       'LeaveCopyEverywhere',
       'LeaveCopyDown',
       'ProbCache',
       'CacheLessForMore',
       'RandomBernoulli',
       'RandomChoice',
       'PopularityBasedCollaborative',
       'CRAN_CountBloomFilter1',
       'Hierarchical_cache',
       'Hierarchical_cache2',
       'Hierarchical_cache_NonCop',
       'Dividing_cache',
       'BF_error_caculate'
       
           ]


@register_strategy('PARTITION')
class Partition(Strategy):
    """Partition caching strategy.

    In this strategy the network is divided into as many partitions as the number
    of caching nodes and each receiver is statically mapped to one and only one
    caching node. When a request is issued it is forwarded to the cache mapped
    to the receiver. In case of a miss the request is routed to the source and
    then returned to cache, which will store it and forward it back to the
    receiver.

    This requires median cache placement, which optimizes the placement of
    caches for this strategy.

    This strategy is normally used with a small number of caching nodes. This
    is the the behaviour normally adopted by Network CDN (NCDN). Google Global
    Cache (GGC) operates this way.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller):
        super(Partition, self).__init__(view, controller)
        if 'cache_assignment' not in self.view.topology().graph:
            raise ValueError('The topology does not have cache assignment '
                             'information. Have you used the optimal median '
                             'cache assignment?')
        self.cache_assignment = self.view.topology().graph['cache_assignment']

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        source = self.view.content_source(content)
        self.controller.start_session(time, receiver, content, log)
        cache = self.cache_assignment[receiver]
        self.controller.forward_request_path(receiver, cache)
        if not self.controller.get_content(cache):
            self.controller.forward_request_path(cache, source)
            self.controller.get_content(source)
            self.controller.forward_content_path(source, cache)
            self.controller.put_content(cache)
        self.controller.forward_content_path(cache, receiver)
        self.controller.end_session()


@register_strategy('EDGE')
class Edge(Strategy):
    """Edge caching strategy.

    In this strategy only a cache at the edge is looked up before forwarding
    a content request to the original source.

    In practice, this is like an LCE but it only queries the first cache it
    finds in the path. It is assumed to be used with a topology where each
    PoP has a cache but it simulates a case where the cache is actually further
    down the access network and it is not looked up for transit traffic passing
    through the PoP but only for PoP-originated requests.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller):
        super(Edge, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        edge_cache = None
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                edge_cache = v
                if self.controller.get_content(v):
                    serving_node = v
                else:
                    # Cache miss, get content from source
                    self.controller.forward_request_path(v, source)
                    self.controller.get_content(source)
                    serving_node = source
                break
        else:
            # No caches on the path at all, get it from source
            self.controller.get_content(v)
            serving_node = v

        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        self.controller.forward_content_path(serving_node, receiver, path)
        if serving_node == source:
            self.controller.put_content(edge_cache)
        self.controller.end_session()


@register_strategy('Non_cache')
class Non_cache(Strategy):
    """Non_cache strategy.

    never cache any content
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(LeaveCopyEverywhere, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            self.controller.get_content(v)
            serving_node = v
            
        serving_node = source
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
        self.controller.end_session()


@register_strategy('LCE')
class LeaveCopyEverywhere(Strategy):
    """Leave Copy Everywhere (LCE) strategy.

    In this strategy a copy of a content is replicated at any cache on the
    path between serving node and receiver.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(LeaveCopyEverywhere, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                self.controller.put_content(v)
        self.controller.end_session()



@register_strategy('LCD')
class LeaveCopyDown(Strategy):
    """Leave Copy Down (LCD) strategy.

    According to this strategy, one copy of a content is replicated only in
    the caching node you hop away from the serving node in the direction of
    the receiver. This strategy is described in [2]_.

    Rereferences
    ------------
    ..[1] N. Laoutaris, H. Che, i. Stavrakakis, The LCD interconnection of LRU
          caches and its analysis.
          Available: http://cs-people.bu.edu/nlaout/analysis_PEVA.pdf
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(LeaveCopyDown, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
            else:
                # No cache hits, get content from source
                self.controller.get_content(v)
                serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        # Leave a copy of the content only in the cache one level down the hit
        # caching node
        copied = False
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if not copied and v != receiver and self.view.has_cache(v):
                self.controller.put_content(v)
                copied = True
        self.controller.end_session()


@register_strategy('PROB_CACHE')
class ProbCache(Strategy):
    """ProbCache strategy [3]_

    This strategy caches content objects probabilistically on a path with a
    probability depending on various factors, including distance from source
    and destination and caching space available on the path.

    This strategy was originally proposed in [2]_ and extended in [3]_. This
    class implements the extended version described in [3]_. In the extended
    version of ProbCache the :math`x/c` factor of the ProbCache equation is
    raised to the power of :math`c`.

    References
    ----------
    ..[2] I. Psaras, W. Chai, G. Pavlou, Probabilistic In-Network Caching for
          Information-Centric Networks, in Proc. of ACM SIGCOMM ICN '12
          Available: http://www.ee.ucl.ac.uk/~uceeips/prob-cache-icn-sigcomm12.pdf
    ..[3] I. Psaras, W. Chai, G. Pavlou, In-Network Cache Management and
          Resource Allocation for Information-Centric Networks, IEEE
          Transactions on Parallel and Distributed Systems, 22 May 2014
          Available: http://doi.ieeecomputersociety.org/10.1109/TPDS.2013.304
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, t_tw=10):
        super(ProbCache, self).__init__(view, controller)
        self.t_tw = t_tw
        self.cache_size = view.cache_nodes(size=True)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        c = len([v for v in path if self.view.has_cache(v)])
        x = 0.0
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            N = sum([self.cache_size[n] for n in path[hop - 1:]
                     if n in self.cache_size])
            if v in self.cache_size:
                x += 1
            self.controller.forward_content_hop(u, v)
            if v != receiver and v in self.cache_size:
                # The (x/c) factor raised to the power of "c" according to the
                # extended version of ProbCache published in IEEE TPDS
                prob_cache = float(N) / (self.t_tw * self.cache_size[v]) * (x / c) ** c
                if random.random() < prob_cache:
                    self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('CL4M')
class CacheLessForMore(Strategy):
    """Cache less for more strategy [4]_.

    This strategy caches items only once in the delivery path, precisely in the
    node with the greatest betweenness centrality (i.e., that is traversed by
    the greatest number of shortest paths). If the argument *use_ego_betw* is
    set to *True* then the betweenness centrality of the ego-network is used
    instead.

    References
    ----------
    ..[4] W. Chai, D. He, I. Psaras, G. Pavlou, Cache Less for More in
          Information-centric Networks, in IFIP NETWORKING '12
          Available: http://www.ee.ucl.ac.uk/~uceeips/centrality-networking12.pdf
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, use_ego_betw=False, **kwargs):
        super(CacheLessForMore, self).__init__(view, controller)
        topology = view.topology()
        if use_ego_betw:
            self.betw = dict((v, nx.betweenness_centrality(nx.ego_graph(topology, v))[v])
                             for v in topology.nodes_iter())
        else:
            self.betw = nx.betweenness_centrality(topology)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        # No cache hits, get content from source
        else:
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        # get the cache with maximum betweenness centrality
        # if there are more than one cache with max betw then pick the one
        # closer to the receiver
        max_betw = -1
        designated_cache = None
        for v in path[1:]:
            if self.view.has_cache(v):
                if self.betw[v] >= max_betw:
                    max_betw = self.betw[v]
                    designated_cache = v
        # Forward content
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if v == designated_cache:
                self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('RAND_BERNOULLI')
class RandomBernoulli(Strategy):
    """Bernoulli random cache insertion.

    In this strategy, a content is randomly inserted in a cache on the path
    from serving node to receiver with probability *p*.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, p=0.2, **kwargs):
        super(RandomBernoulli, self).__init__(view, controller)
        self.p = p

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if v != receiver and self.view.has_cache(v):
                if random.random() < self.p:
                    self.controller.put_content(v)
        self.controller.end_session()

@register_strategy('RAND_CHOICE')
class RandomChoice(Strategy):
    """Random choice strategy

    This strategy stores the served content exactly in one single cache on the
    path from serving node to receiver selected randomly.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(RandomChoice, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        caches = [v for v in path[1:-1] if self.view.has_cache(v)]
        designated_cache = random.choice(caches) if len(caches) > 0 else None
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if v == designated_cache:
                self.controller.put_content(v)
        self.controller.end_session()

@register_strategy('PBC')
class PopularityBasedCollaborative(Strategy):
    """Popularity Based Collaborative Strategy
    
    """
    
    @inheritdoc(Strategy)
    def __init__(self, view, controller, use_global_popularity,**kwargs):
        super(PopularityBasedCollaborative, self).__init__(view, controller)
        self.use_global_popularity = use_global_popularity
        
        if(not use_global_popularity):
            self.content_popularity_index = {} # map node with content index dic
            #use all nodes or has cache nodes?
            for i in view.topology().nodes():
                self.content_popularity_index[i] = {}
        
    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        compareTag = True
        cacheNode = []
        self.controller.start_session(time, receiver, content, log)
        
        if(self.use_global_popularity):
            for u, v in path_links(path):
                self.controller.forward_request_hop(u, v)
                if self.view.has_cache(v):
                    if self.controller.get_content(v):
                        serving_node = v
                        break
                    else:
                        if self.view.is_cache_full(v):
                            if compareTag:
                                for cache in self.view.cache_dump(v):
                                    if(content < cache):
                                        cacheNode.append(v)
                                        compareTag = False
                                        break
                        else:
                            if compareTag:
                                cacheNode.append(v)
                                compareTag = False
                # No cache hits, get content from source
                self.controller.get_content(v)
                serving_node = v
            # Return content
            path = list(reversed(self.view.shortest_path(receiver, serving_node)))
            for u, v in path_links(path):
                self.controller.forward_content_hop(u, v)
                if self.view.has_cache(v):
                    # insert content
                    if v in cacheNode:
                        self.controller.put_content(v)
        else:
            for u, v in path_links(path):
                self.controller.forward_request_hop(u, v)
                if self.view.has_cache(v):
                    if(content in self.content_popularity_index[v]):
                        self.content_popularity_index[v][content] = self.content_popularity_index[v][content] + 1
                    else:
                        self.content_popularity_index[v][content] = 1
                        
                    if self.controller.get_content(v):
                        serving_node = v
                        break
                    else:
                        if self.view.is_cache_full(v):
                            if compareTag:
                                contentIndex = self.content_popularity_index[v][content]
                                for cache in self.view.cache_dump(v):
                                    if(contentIndex > self.content_popularity_index[v][cache]):
                                        cacheNode.append(v)
                                        compareTag = False
                                        break
                        else:
                            if compareTag:
                                cacheNode.append(v)
                                compareTag = False
                # No cache hits, get content from source
                self.controller.get_content(v)
                serving_node = v
            # Return content
            path = list(reversed(self.view.shortest_path(receiver, serving_node)))
            for u, v in path_links(path):
                self.controller.forward_content_hop(u, v)
                if self.view.has_cache(v):
                    # insert content
                    if v in cacheNode:
                        self.controller.put_content(v)
        self.controller.end_session()

        
@register_strategy('CRANBF2')
class CRAN_CountBloomFilter2(Strategy):
    """CRAN_CountBloomFilter (CRANBF2) strategy.

    In this strategy a copy of a content is replicated at any cache on the
    path between serving node and receiver.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(CRAN_CountBloomFilter2, self).__init__(view, controller)
        self.cbf = {}
        for i in view.topology().nodes():
            self.cbf[i] = example.init()

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        mark = 0
        cacheNode = []
        tag = True
        k = 3
        cooperation = False
        contentstr = str(content)
        length = len(contentstr)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            cbf = self.cbf[v]
            if(not example.insert(cbf, contentstr, length)):
                print ('insert failed')
            if self.view.has_cache(v):
                    if self.controller.get_content(v):
                        serving_node = v
                        break
                    else:
                            if tag:
                                #contentIndex = example.multiplicityquery(cbf, contentstr, length)
                                #for cache in self.view.cache_dump(v):
                                    #cachestr = str(cache)
                                    #len_cache = len(cachestr)
                                    #if(contentIndex > example.multiplicityquery(cbf, cachestr, len_cache)):
                                        #cacheNode.append(v)
                                        #tag = False
                                        #break
                                #print(len(self.view.cache_dump(v)))
                                #print("node: %d" %v)
                                if len(self.view.cache_dump(v)) <= 40:
                                        cacheNode.append(v)
                                        tag = False
                                for cache in self.view.cache_dump(v):
                                        if content < cache :
                                                cacheNode.append(v)
                                                tag = False
                                                break
            if u == receiver:
                mark = 1
            elif (mark == 1 and u != receiver):
                mark = 0
                for son in range(v*k+1,(v+1)*k+1):
                    if self.view.cache_lookup(son, content):
                        if self.controller.get_content(son):
                            serving_node = son
                            break
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                if v in cacheNode:
                    self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('CRANBF1')
class CRAN_CountBloomFilter1(Strategy):
    """CRAN_CountBloomFilter (CRANBF2) strategy.

    In this strategy a copy of a content is replicated at any cache on the
    path between serving node and receiver.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(CRAN_CountBloomFilter1, self).__init__(view, controller)
        self.cbf = {}
        for i in view.topology().nodes():
            self.cbf[i] = example.init()

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        mark = 0
        cacheNode = []
        tag = True
        k = 2
        cooperation = False
        contentstr = str(content)
        length = len(contentstr)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            cbf = self.cbf[v]
            if(not example.insert(cbf, contentstr, length)):
                print ('insert failed')
            if self.view.has_cache(v):
                    if self.controller.get_content(v):
                        serving_node = v
                        break
                    else:
                        if self.view.is_cache_full(v):
                            if tag:
                                for cache in self.view.cache_dump(v):
                                        if content < cache :
                                                cacheNode.append(v)
                                                tag = False
                                                break
                        else:
                            cacheNode.append(v)
                            tag = False
            if u == receiver:
                mark = 1
            elif (mark == 1 and u != receiver):
                mark = 0
                for son in range(v*k+1,(v+1)*k+1):
                    if son != u:
                        if self.view.cache_lookup(son, content):
                            if self.controller.get_content(son):
                                serving_node = son
                                cooperation = True
                                break
            # No cache hits, get content from source
            if cooperation:
                break
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                if v in cacheNode and not cooperation:
                    self.controller.put_content(v)
                    if u != 0:
                        self.controller.remove_content(u)
        self.controller.end_session()
        

@register_strategy('HC')
class Hierarchical_cache(Strategy):
    """Hierarchical_cache(Hierarchical) strategy.
    Edge popular & Coexistence
    In this strategy a copy of a content is replicated at any cache on the
    path between serving node and receiver.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Hierarchical_cache, self).__init__(view, controller)
        self.popularity_index = {}
        for i in view.topology().nodes():
            self.popularity_index[i] = {}

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        mark = 0
        cacheNode = []
        tag = True
        k = 4
        cooperation = False
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if(content in self.popularity_index[v]):
                    self.popularity_index[v][content] = self.popularity_index[v][content] + 1
                else:
                    self.popularity_index[v][content] = 1  
                    
                if self.controller.get_content(v):
                    serving_node = v
                    break
                

                else:
                    if v != source:
                        for son in range(v*k+1,(v+1)*k+1):
                            if son != u and self.view.has_cache(son):
                                if self.view.cache_lookup(son, content):
                                    if self.controller.get_content(son):
                                        serving_node = son
                                        cooperation = True
                                        break
                        if cooperation:
                            break
                    
                    if self.view.is_cache_full(v):
                        for cache in self.view.cache_dump(v):
                            if self.popularity_index[v][content] > self.popularity_index[v][cache] :
                                if tag:
                                    cacheNode.append(v)
                                    tag = False
                                    break
                    else:
                        if tag:
                            cacheNode.append(v)
                            tag = False
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v

        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                if v in cacheNode and not cooperation:
                    self.controller.put_content(v)
        self.controller.end_session()
          

@register_strategy('HC2')
class Hierarchical_cache2(Strategy):
    """Hierarchical_cache2(Hierarchical) strategy.
    Core popular & Coexistence
    implement local popularity and cooperation
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Hierarchical_cache2, self).__init__(view, controller)
        self.popularity_index = {}
        for i in view.topology().nodes():
            self.popularity_index[i] = {}

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        mark = 0
        cacheNode = []
        tag = True
        k = 3
        cooperation = False
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if(content in self.popularity_index[v]):
                    self.popularity_index[v][content] = self.popularity_index[v][content] + 1
                else:
                    self.popularity_index[v][content] = 1    


                if self.controller.get_content(v):
                    serving_node = v
                    break
                else:
                    if v != source and self.view.has_cache(u):
                        for son in range(v*k+1,(v+1)*k+1):
                            if son != u and self.view.has_cache(son):
                                if self.view.cache_lookup(son, content):
                                    if self.controller.get_content(son):
                                        serving_node = son
                                        cooperation = True
                                        break
                        if cooperation:
                            break
                    
                    if self.view.is_cache_full(v):
                        for cache in self.view.cache_dump(v):
                            if self.popularity_index[v][content] > self.popularity_index[v][cache] :
                                cacheNode.append(v)
                                break
                    else:
                        cacheNode.append(v)
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                if v in cacheNode and serving_node == source and not cooperation:
                    if tag:
                        self.controller.put_content(v)
                        tag = False
        self.controller.end_session()


@register_strategy('HC3')
class Hierarchical_cache3(Strategy):
    """Hierarchical_cache(Hierarchical) strategy.
    Edge popular & Non-Coexistence
    In this strategy a copy of a content is replicated at any cache on the
    path between serving node and receiver.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Hierarchical_cache3, self).__init__(view, controller)
        self.popularity_index = {}
        for i in view.topology().nodes():
            self.popularity_index[i] = {}

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        mark = 0
        cacheNode = []
        tag = True
        k = 3
        cooperation = False
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if(content in self.popularity_index[v]):
                    self.popularity_index[v][content] = self.popularity_index[v][content] + 1
                else:
                    self.popularity_index[v][content] = 1  
                    
                if self.controller.get_content(v):
                    serving_node = v
                    break
                

                else:
                    if v != source:
                        for son in range(v*k+1,(v+1)*k+1):
                            if son != u and self.view.has_cache(son):
                                if self.view.cache_lookup(son, content):
                                    if self.controller.get_content(son):
                                        serving_node = son
                                        cooperation = True
                                        break
                        if cooperation:
                            break
                    
                    if self.view.is_cache_full(v):
                        for cache in self.view.cache_dump(v):
                            if self.popularity_index[v][content] > self.popularity_index[v][cache] :
                                if tag:
                                    cacheNode.append(v)
                                    tag = False
                                    break
                    else:
                        if tag:
                            cacheNode.append(v)
                            tag = False
                            
                            
                    if cooperation:
                        break
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v

        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                if v in cacheNode:
                    self.controller.put_content(v)
        self.controller.end_session()

        
@register_strategy('HC_NonCop')
class Hierarchical_cache_NonCop(Strategy):
    """Hierarchical_cache2(Hierarchical) strategy.

    implement local popularity and cooperation
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Hierarchical_cache_NonCop, self).__init__(view, controller)
        self.popularity_index = {}
        self.bloomfilter = {}
        for i in view.topology().nodes():
            self.popularity_index[i] = {}
            self.bloomfilter[i] = BloomFilter(62530, 4)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        mark = 0
        cacheNode = []
        k = 3
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if(content in self.popularity_index[v]):
                    self.popularity_index[v][content] = self.popularity_index[v][content] + 1
                else:
                    self.popularity_index[v][content] = 1


            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
                else:
                    if self.view.is_cache_full(v):
                        for cache in self.view.cache_dump(v):
                            if self.popularity_index[v][content] > self.popularity_index[v][cache] :
                                cacheNode.append(v)
                                break
                    else:
                        cacheNode.append(v)
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                if v in cacheNode:
                    self.controller.put_content(v)
                    tag = False
        self.controller.end_session()


@register_strategy('DC')
class Dividing_cache(Strategy):
    """Hierarchical_cache3(Hierarchical) strategy.

    implement local popularity and cooperation
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Dividing_cache, self).__init__(view, controller)
        self.popularity_index = {}
        self.popcache_num = {}
        self.privatecache_num = {}
        for i in view.topology().nodes():
            self.popularity_index[i] = {}

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        mark = False
        cacheNode = []
        tag = True
        k = 4
        cooperation = False
        for u, v in path_links(path):
            count = 0
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if(content in self.popularity_index[v]):
                    self.popularity_index[v][content] = self.popularity_index[v][content] + 1
                else:
                    self.popularity_index[v][content] = 1    


                if self.controller.get_content(v):
                    serving_node = v
                    break
                else:
                    
                    if v != source and self.view.has_cache(u):
                        for son in range(v*k+1,(v+1)*k+1):
                            if self.view.cache_lookup(son, content):
                                if self.controller.get_content(son):
                                    serving_node = son
                                    cooperation = True
                                    break
                    
                    # View the popularity ranking number
                    if self.view.is_cache_full(v):
                        if v != source and self.view.has_cache(u):
                            for cache in self.view.cache_dump(v):
                                if self.popularity_index[v][content] > self.popularity_index[v][cache] and not cooperation:
                                    if tag:
                                        cacheNode.append(v)
                                    break
                        else:
                            for cache in self.view.cache_dump(v):
                                if self.popularity_index[v][content] > self.popularity_index[v][cache]:
                                    count = count + 1
                        
                        #upper half cache by popularity
                            if count > self.view.cache_size(v)*2/3:
                                if tag:
                                    cacheNode.append(v)
                                    tag = False
                        #lower half cache when not cooperate        
                            elif count > 0:
                                if tag:
                                    cacheNode.append(v)
                                    tag = False
                                    mark = True
                                                    
                    else:
                        if tag:
                            cacheNode.append(v)
                            tag = False
                            
                    
                    if cooperation:
                        break
                    
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                if v in cacheNode:
                    if cooperation and mark:
                        continue
                    else:
                        self.controller.put_content(v)
        self.controller.end_session()

@register_strategy('DC_TREE')
class Dividing_cache_TREE(Strategy):
    """Dividing_cache_TREE(Hierarchical) strategy.

    implement local popularity and cooperation
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Dividing_cache_TREE, self).__init__(view, controller)
        self.popularity_index = {}
        self.popcache_num = {}
        self.privatecache_num = {}
        for i in view.topology().nodes():
            self.popularity_index[i] = {}
            
    def iter_query(self, node, k, n, content):
        for son in range(node*k+1,(node+1)*k+1):
            if self.view.cache_lookup(son, content):
                if self.controller.get_content(son):
                    return son
        if node > n:
            return
        for son in range(node*k+1,(node+1)*k+1):
            res = self.iter_query(son, k, n, content)
            if res:
                return res
        return       

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        cached_tag = True
        cacheNode = []
        tag = True
        k = 3
        cooperation = False
        for u, v in path_links(path):
            count = 0
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if(content in self.popularity_index[v]):
                    self.popularity_index[v][content] = self.popularity_index[v][content] + 1
                else:
                    self.popularity_index[v][content] = 1    


                if self.controller.get_content(v):
                    serving_node = v
                    break
                else:
                    
                    if v != source and self.view.has_cache(u):
                        res = self.iter_query(v, k, 3, content)
                        if res:
                            serving_node = res
                            cooperation = True
                    
                    # View the popularity ranking number
                    if self.view.is_cache_full(v):
                        
                        if v < 4 and v > 0: # level 1 node depth = 1
                            for cache in self.view.cache_dump(v):
                                if self.popularity_index[v][content] > self.popularity_index[v][cache] and not cooperation:
                                    if tag:
                                        cacheNode.append(v)
                                        break
                        else: #edge node depth = 3
                            for cache in self.view.cache_dump(v):
                                if self.popularity_index[v][content] > self.popularity_index[v][cache]:
                                    count = count + 1
                        
                            #upper half cache by popularity
                            if count > self.view.cache_size(v)/2:
                                if tag:
                                    cacheNode.append(v)
                                    tag = False
                            #lower half cache when not cooperate        
                            elif count > 0:
                                if tag:
                                    cacheNode.append(v)
                          
                                                    
                    else:
                        if tag:
                            cacheNode.append(v)
                            tag = False
                            
                    
                    if cooperation:
                        break
                    
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                if v in cacheNode:
                    if not tag:
                        self.controller.put_content(v)
                    elif serving_node == 0:
                        if cached_tag:
                            self.controller.put_content(v)
                            cached_tag = False
        self.controller.end_session()
        

@register_strategy('HC_TREE')
class Hierarchical_cache_TREE(Strategy):
    """Hierarchical_cache_TREE(Hierarchical) strategy.
    Core popular & Coexistence
    implement local popularity and cooperation
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Hierarchical_cache_TREE, self).__init__(view, controller)
        self.popularity_index = {}
        for i in view.topology().nodes():
            self.popularity_index[i] = {}

    def iter_query(self, node, k, n, content):
        for son in range(node*k+1,(node+1)*k+1):
            if self.view.cache_lookup(son, content):
                if self.controller.get_content(son):
                    return son
        if node > n:
            return
        for son in range(node*k+1,(node+1)*k+1):
            res = self.iter_query(son, k, n, content)
            if res:
                return res
        return       


    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        mark = 0
        cacheNode = []
        tag = True
        k = 3
        cooperation = False
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if(content in self.popularity_index[v]):
                    self.popularity_index[v][content] = self.popularity_index[v][content] + 1
                else:
                    self.popularity_index[v][content] = 1    


                if self.controller.get_content(v):
                    serving_node = v
                    break
                
                else:
                    
                    if v != source and self.view.has_cache(u):
                        res = self.iter_query(v, k, 3, content)
                        if res:
                            serving_node = res
                            cooperation = True
                        if cooperation:
                            break
                    
                    if self.view.is_cache_full(v):
                        for cache in self.view.cache_dump(v):
                            if self.popularity_index[v][content] > self.popularity_index[v][cache] :
                                if tag:
                                    cacheNode.append(v)
                                    tag = False
                                    break
                    else:
                        cacheNode.append(v)
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                if v in cacheNode and not cooperation:
                    self.controller.put_content(v)
        self.controller.end_session()
        
@register_strategy('BF_ERROR')
class BF_error_caculate(Strategy):
    """BF_error_caculate
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(BF_error_caculate, self).__init__(view, controller)
        self.popularity_index = {}
        self.bloomfilter = {}
        self.noncop = 0
        self.error = 0
        self.ratio = 0
        self.count = 1
        self.count2 = 1
        self.cachesize = view.topology().node[5]['stack'][1]['cache_size']
        for i in view.topology().nodes():
            self.popularity_index[i] = {}
            self.bloomfilter[i] = BloomFilter(62530, 4)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        mark = 0
        cacheNode = []
        tag = True
        k = 3
        cooperation = False
        self.count += 1
        self.count2 += 1
        
        if self.count2 > 10:
            self.count2 = 0
            for i in range(13, 40):
                self.bloomfilter[i] = BloomFilter(62530, 4)
                for cache in self.view.cache_dump(i):
                    self.bloomfilter[i].add(str(cache))
        
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if(content in self.popularity_index[v]):
                    self.popularity_index[v][content] = self.popularity_index[v][content] + 1
                else:
                    self.popularity_index[v][content] = 1  
                    
                if self.controller.get_content(v):
                    serving_node = v
                    break
                

                else:
                    if v != source:
                        
                        if self.count2 > 10 and u != receiver:
                            self.count2 = 0
                            for i in range(v*k+1,(v+1)*k+1):
                                self.bloomfilter[i] = BloomFilter(8*self.cachesize, 4)
                                if self.view.cache_dump(i):
                                    for cache in self.view.cache_dump(i):
                                        self.bloomfilter[i].add(str(cache))
                                        
                        for son in range(v*k+1,(v+1)*k+1):
                            if son != u and self.view.has_cache(son):
                                if self.view.cache_lookup(son, content):
                                    cooperation = True
                        if not cooperation and self.view.has_cache(u):
                            self.noncop += 1         
                            for son in range(v*k+1,(v+1)*k+1):
                                if son != u and str(content) in self.bloomfilter[son]:
                                    self.error += 1
                                    break
                        
                        for son in range(v*k+1,(v+1)*k+1):
                            if son != u and self.view.has_cache(son):
                                if self.view.cache_lookup(son, content):
                                    if self.controller.get_content(son):
                                        serving_node = son
                                        cooperation = True
                                        break
                        if cooperation:
                            break
                    
                    if self.view.is_cache_full(v):
                        for cache in self.view.cache_dump(v):
                            if self.popularity_index[v][content] > self.popularity_index[v][cache] :
                                if tag:
                                    cacheNode.append(v)
                                    tag = False
                                    break
                    else:
                        if tag:
                            cacheNode.append(v)
                            tag = False
                            
                            
                    if cooperation:
                        break
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        self.ratio = 1.0*self.error/(self.error + self.correct)
        if self.count > 100:
            print(self.ratio)
            self.count = 0
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                if v in cacheNode:
                    self.controller.put_content(v)
                    self.bloomfilter[v].add(str(content))
        self.controller.end_session()
        

@register_strategy('Pop_COUNT')
class Pop_count(Strategy):
    """Pop_count strategy.

    never cache any content
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Pop_count, self).__init__(view, controller)
        self.count = [0 for i in range(0, 100001)]
        self.map = [0 for i in range(0, 100001)]
        self.popcount = 0

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        
        self.count[content] += 1
        if self.count[content] > 4 and self.map[content] == 0:
            self.map[content] = 1
            self.popcount += 1
            print self.popcount
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            self.controller.get_content(v)
            serving_node = v
            
        serving_node = source
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
        self.controller.end_session()
        
        
@register_strategy('DC_ALL')
class Dividing_cache_all(Strategy):
    """Dividing_cache_all(Hierarchical) strategy.

    implement local popularity and cooperation
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Dividing_cache_all, self).__init__(view, controller)
        self.popularity_index = {}
        self.popcache_num = {}
        self.privatecache_num = {}
        self.next_index = {}
        print(view.topology().nodes())
        for i in view.topology().nodes():
            self.popularity_index[i] = {}
        for i in view.topology().nodes():
            self.next_index[i] = []
        for i in view.topology().nodes():
            if self.view.has_cache(i):
                for j in view.topology().nodes():
                    if self.view.has_cache(j):
                        route = len(self.view.shortest_path(i, j))
                        if route == 2:
                            self.next_index[i].append(j)


    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        cacheNode = []
        tag = True
        cooperation = False
        for u, v in path_links(path):
            count = 0
            self.controller.forward_request_hop(u, v)
            if content in self.popularity_index[v]:
                self.popularity_index[v][content] = self.popularity_index[v][content] + 1
            else:
                self.popularity_index[v][content] = 1

            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
                else:
                    for i in self.next_index[v]:
                        if self.view.cache_lookup(i, content):
                            if self.controller.get_content(i):
                                serving_node = i
                                cooperation = True
                                break

                    if self.view.is_cache_full(v):

                        for cache in self.view.cache_dump(v):
                                if cache in self.popularity_index[v]:
                                        pass
                                else:
                                        self.popularity_index[v][cache] = 1
                                if content in self.popularity_index[v]:
                                        pass
                                else:
                                        self.popularity_index[v][content] = 1

                                if self.popularity_index[v][content] > self.popularity_index[v][cache]:
                                        count = count + 1

                        #upper half cache by popularity
                        if count > self.view.cache_size(v)*2/3:
                            if tag:
                                cacheNode.append(v)
                                tag = False

                    if cooperation:
                        break

            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):

                                    # View the popularity ranking number
                    if self.view.is_cache_full(v):

                        if v in cacheNode:
                            self.controller.put_content(v)
                            continue
                        for cache in self.view.cache_dump(v):
                            if cache in self.popularity_index[v]:
                                pass
                            else:
                                self.popularity_index[v][cache] = 1
                            if content in self.popularity_index[v]:
                                pass
                            else:
                                self.popularity_index[v][content] = 1
                            if self.popularity_index[v][content] > self.popularity_index[v][cache]:
                                count = count + 1

                        #upper half cache by popularity
                        if count > self.view.cache_size(v)*2/3:
                            if tag:
                                self.controller.put_content(v)
                                tag = False
                        #lower half cache when not cooperate        
                        elif count > 0:
                            if not cooperation:
                                self.controller.put_content(v)
                                cooperation = True
                                                    
                    else:
                        self.controller.put_content(v)
                
        self.controller.end_session()
        
        
        
        
@register_strategy('CC')
class Collaborative_cache(Strategy):
    """Collaborative_cache(Hierarchical) strategy.

    implement local popularity and cooperation
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Collaborative_cache, self).__init__(view, controller)
        self.popcache_num = {}
        self.privatecache_num = {}
        self.next_index = {}
        for i in view.topology().nodes():
            self.next_index[i] = []
        for i in view.topology().nodes():
            if self.view.has_cache(i):
                for j in view.topology().nodes():
                    if self.view.has_cache(j):   
                        route = len(self.view.shortest_path(i, j))
                        if route == 2:
                            self.next_index[i].append(j)
                            

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        tag = True
        cooperation = False
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
                else:
                    for i in self.next_index[v]:
                        if self.view.cache_lookup(i, content):
                            if self.controller.get_content(i):
                                serving_node = i
                                cooperation = True
                                break
                    
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                
                                    # View the popularity ranking number
                    if self.view.is_cache_full(v):
                        
                        if not cooperation:
                            self.controller.put_content(v)
                            cooperation = True
                                                    
                    else:
                        self.controller.put_content(v)
                
        self.controller.end_session()
        
        
        

@register_strategy('DC_ALL_BF')
class Dividing_cache_all_BF(Strategy):
    """Dividing_cache_all_BF(Hierarchical) strategy.

    implement local popularity and cooperation
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Dividing_cache_all_BF, self).__init__(view, controller)
        self.popularity_index = {}
        self.bloomfilter = {}
        self.popcache_num = {}
        self.privatecache_num = {}
        self.next_index = {}
        self.degree = {}
        self.ratio = 1
        self.error = 0
        self.ori = 0
        self.bfhit = 1
        self.cachesize = 0
        for i in view.topology().nodes():
            if self.view.has_cache(i):
                self.cachesize = view.topology().node[i]['stack'][1]['cache_size']
        print self.cachesize
        for i in view.topology().nodes():
            self.popularity_index[i] = {}
            self.bloomfilter[i] = BloomFilter(62530, 4)    
        
        for i in view.topology().nodes():
            self.next_index[i] = []
            self.degree[i] = 0
        for i in view.topology().nodes():
            if self.view.has_cache(i):
                for j in view.topology().nodes():
                    if self.view.has_cache(j):
                        route = len(self.view.shortest_path(i, j))
                        if route == 2:
                            self.next_index[i].append(j)
                            self.degree[i] += 1


    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        cacheNode = []
        tag = True
        cooperation = False
        

        for i in view.topology().nodes():
            if self.view.has_cache(i):
                self.bloomfilter[i] = BloomFilter(62530, 4)
                for cache in self.view.cache_dump(i):
                    self.bloomfilter[i].add(str(cache))
        
        for u, v in path_links(path):
            count = 0
            self.controller.forward_request_hop(u, v)
            if content in self.popularity_index[v]:
                self.popularity_index[v][content] = self.popularity_index[v][content] + 1
            else:
                self.popularity_index[v][content] = 1

            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
                else:
                    
                    for i in self.next_index[v]:
                        if self.view.cache_lookup(i, content):
                            if self.controller.get_content(i):
                                serving_node = i
                                cooperation = True
                                break
                            
                    if not cooperation and self.view.has_cache(u):
                        self.noncop += 1         
                        for i in self.next_index[v]:
                            if i != u and str(content) in self.bloomfilter[i]:
                                self.error += 1
                                break



                    if self.view.is_cache_full(v):

                        for cache in self.view.cache_dump(v):
                                if cache in self.popularity_index[v]:
                                        pass
                                else:
                                        self.popularity_index[v][cache] = 1
                                if content in self.popularity_index[v]:
                                        pass
                                else:
                                        self.popularity_index[v][content] = 1

                                if self.popularity_index[v][content] > self.popularity_index[v][cache]:
                                        count = count + 1

                        #upper half cache by popularity
                        if count > self.view.cache_size(v)*1/11:
                            if tag:
                                cacheNode.append(v)
                                tag = False

                    if cooperation:
                        break

            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        self.ratio = 1.0*self.error/(self.error + self.correct)
        if self.count > 100:
            print(self.ratio)
            self.count = 0
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):

                                    # View the popularity ranking number
                    if self.view.is_cache_full(v):

                        if v in cacheNode:
                            self.controller.put_content(v)
                            continue
                        for cache in self.view.cache_dump(v):
                            if cache in self.popularity_index[v]:
                                pass
                            else:
                                self.popularity_index[v][cache] = 1
                            if content in self.popularity_index[v]:
                                pass
                            else:
                                self.popularity_index[v][content] = 1
                            if self.popularity_index[v][content] > self.popularity_index[v][cache]:
                                count = count + 1

                        #upper half cache by popularity
                        if count > self.view.cache_size(v)*2/3:
                            if tag:
                                self.controller.put_content(v)
                                tag = False
                        #lower half cache when not cooperate        
                        elif count > 0:
                            if not cooperation:
                                self.controller.put_content(v)
                                cooperation = True
                                                    
                    else:
                        self.controller.put_content(v)
                
        self.controller.end_session()
                
        
        
@register_strategy('SC')
class Source_cache(Strategy):
    """Source Cache strategy.

    implement local popularity and cooperation
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Source_cache, self).__init__(view, controller)
        self.ridmap = []
        self.popularity_index = {}
        for i in self.view.topology().nodes():
            self.ridmap.append({})
            self.popularity_index[i] = {}                

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        mark = 0
        cacheNode = []
        cooperation = False
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            
            count = 0
            if self.view.has_cache(v):

                if v != source:
                    if not self.view.is_cache_full(v):
                        cacheNode.append(v)
                    
                    if self.controller.get_content(v):
                        serving_node = v
                        
                        if(content in self.popularity_index[v]):
                            self.popularity_index[v][content] = self.popularity_index[v][content] + 1
                        else:
                            self.popularity_index[v][content] = 1  
                        
                        if content in self.ridmap[v]:
                            self.ridmap[v][content].append((receiver, time))
                        else:
                            self.ridmap[v][content] = [(receiver, time)]
                        
                        for i in self.ridmap[v][content]:
                            if timetag - i[1] > 1:
                                self.ridmap[v][content].remove(i)
                                
                        count = len(self.ridmap[v][content])
                        p = random.randint(1, 10)
                        if self.popularity_index[v][content] > p: 
                            if self.view.has_cache(u):
                                cacheNode.append(u)
                        
                        break
                
                else:
                    if self.view.has_cache(u):
                        cacheNode.append(u)

                # No cache hits, get content from source
            else:    
                self.controller.get_content(v)
                serving_node = v

        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                if v in cacheNode:
                    self.controller.put_content(v)
        self.controller.end_session()
        
        
@register_strategy('SCR')
class Source_route_cache(Strategy):
    """Source Cache strategy.

    implement local popularity and cooperation
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Source_route_cache, self).__init__(view, controller)
        self.ridmap = []
        self.popularity_index = {}
        for i in self.view.topology().nodes():
            self.ridmap.append({})
            self.popularity_index[i] = {}                

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        res = {}
        for i in self.view.topology().nodes():
            if self.view.cache_lookup(i, content):
                res[i] = len(self.view.shortest_path(receiver, i))
                pathlen = res[i]
        for i in res:
            if res[i] < pathlen:
                pathlen = res[i]
                source = i
        
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        mark = 0
        cacheNode = []
        cooperation = False
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            
            count = 0
            if self.view.has_cache(v):

                if v != source:
                    if not self.view.is_cache_full(v):
                        cacheNode.append(v)
                    
                    if self.controller.get_content(v):
                        serving_node = v
                        
                        if(content in self.popularity_index[v]):
                            self.popularity_index[v][content] = self.popularity_index[v][content] + 1
                        else:
                            self.popularity_index[v][content] = 1  
                        
                        if content in self.ridmap[v]:
                            self.ridmap[v][content].append((receiver, time))
                        else:
                            self.ridmap[v][content] = [(receiver, time)]
                        
                        for i in self.ridmap[v][content]:
                            if time - i[1] > 1:
                                self.ridmap[v][content].remove(i)
                                
                        count = len(self.ridmap[v][content])
                        p = random.randint(1, 10)
                        if count > p:
                            if self.view.has_cache(u):
                                cacheNode.append(u)

                        
                        break
                
                else:
                    if self.view.has_cache(u):
                        cacheNode.append(u)

                # No cache hits, get content from source
               
            self.controller.get_content(v)
            serving_node = v

        # Return content
        
        copied = False
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if not copied and v != receiver and self.view.has_cache(v):
                self.controller.put_content(v)
                copied = True
        self.controller.end_session()
        

@register_strategy('SCR_PROB')
class Source_route_cache_prob(Strategy):
    """Source Cache strategy.

    implement local popularity and cooperation
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Source_route_cache_prob, self).__init__(view, controller)
        self.ridmap = []
        self.popularity_index = {}
        for i in self.view.topology().nodes():
            self.ridmap.append({})
            self.popularity_index[i] = {}                

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        res = {}
        for i in self.view.topology().nodes():
            if self.view.cache_lookup(i, content):
                res[i] = len(self.view.shortest_path(receiver, i))
                pathlen = res[i]
        for i in res:
            if res[i] < pathlen:
                pathlen = res[i]
                source = i
        
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        copied = True
        pathlen = 0
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            
            pathlen += 1
            
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                        
                    if(content in self.popularity_index[v]):
                        self.popularity_index[v][content] = self.popularity_index[v][content] + 1
                    else:
                        self.popularity_index[v][content] = 1  
                            
                    p = 1
                    if self.popularity_index[v][content] > p:
                        if self.view.has_cache(u):
                            copied = False
                        
                    break
                

                # No cache hits, get content from source
               
            self.controller.get_content(v)
            serving_node = v

        # Return content
        if v == source:
            copied = False

        hop = 0
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            hop += 1
            self.controller.forward_content_hop(u, v)
            if not copied and v != receiver and self.view.has_cache(v):
                k = random.randint(0, pathlen)
                if hop >= k: 
                    self.controller.put_content(v)
                    copied = True
        self.controller.end_session()

@register_strategy('PC')
class Hierarchical_cache_NonCop(Strategy):
    """Hierarchical_cache2(Hierarchical) strategy.

    implement local popularity and cooperation
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Hierarchical_cache_NonCop, self).__init__(view, controller)
        self.popularity_index = {}
        self.bloomfilter = {}
        for i in view.topology().nodes():
            self.popularity_index[i] = {}
            self.bloomfilter[i] = BloomFilter(62530, 4)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        cacheNode = []
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if(content in self.popularity_index[v]):
                    self.popularity_index[v][content] = self.popularity_index[v][content] + 1
                else:
                    self.popularity_index[v][content] = 1


            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
                else:
                    if self.view.is_cache_full(v):
                        for cache in self.view.cache_dump(v):
                            if self.popularity_index[v][content] > self.popularity_index[v][cache] :
                                cacheNode.append(v)
                                break
                    else:
                        cacheNode.append(v)
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                if v in cacheNode:
                    self.controller.put_content(v)
                    tag = False
        self.controller.end_session()

        
@register_strategy('MYLCD')
class MyLeaveCopyDown(Strategy):

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(MyLeaveCopyDown, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        res = {}
        for i in view.topology().nodes():
            if self.view.cache_lookup(i, content):
                res[i] = len(self.view.shortest_path(receiver, i))
                pathlen = res[i]
        for i in res:
            if res[i] < pathlen:
                pathlen = res[i]
                source = i
        
        #print("source : ", i)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
            else:
                # No cache hits, get content from source
                self.controller.get_content(v)
                serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        # Leave a copy of the content only in the cache one level down the hit
        # caching node
        copied = False
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if not copied and v != receiver and self.view.has_cache(v):
                self.controller.put_content(v)
                copied = True
        self.controller.end_session()
        
        
@register_strategy('SCR1')
class Source_route_cache1(Strategy):
    """Source Cache strategy.

    implement local popularity and cooperation
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Source_route_cache1, self).__init__(view, controller)
        self.ridmap = []
        self.popularity_index = {}
        for i in self.view.topology().nodes():
            self.ridmap.append({})
            self.popularity_index[i] = {}                

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        res = {}
        for i in self.view.topology().nodes():
            if self.view.cache_lookup(i, content):
                res[i] = len(self.view.shortest_path(receiver, i))
                pathlen = res[i]
        for i in res:
            if res[i] < pathlen:
                pathlen = res[i]
                source = i
        
        
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        copied = True
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                        
                    if(content in self.popularity_index[v]):
                        self.popularity_index[v][content] = self.popularity_index[v][content] + 1
                    else:
                        self.popularity_index[v][content] = 1  
                            
                    p = 1
                    if self.popularity_index[v][content] > p:
                        if self.view.has_cache(u):
                            copied = False
                        
                    break
                

                # No cache hits, get content from source
               
            self.controller.get_content(v)
            serving_node = v

        # Return content
        if v == source:
            copied = False
        #copied = False
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if not copied and v != receiver and self.view.has_cache(v):
                self.controller.put_content(v)
                copied = True
        self.controller.end_session()
        
@register_strategy('SCR3')
class Source_route_cache3(Strategy):
    """Source Cache strategy.

    implement local popularity and cooperation
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Source_route_cache3, self).__init__(view, controller)
        self.ridmap = []
        self.popularity_index = {}
        for i in self.view.topology().nodes():
            self.ridmap.append({})
            self.popularity_index[i] = {}                

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        res = {}
        for i in self.view.topology().nodes():
            if self.view.cache_lookup(i, content):
                res[i] = len(self.view.shortest_path(receiver, i))
                pathlen = res[i]
        for i in res:
            if res[i] < pathlen:
                pathlen = res[i]
                source = i
        
        
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        copied = True
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                        
                    if(content in self.popularity_index[v]):
                        self.popularity_index[v][content] = self.popularity_index[v][content] + 1
                    else:
                        self.popularity_index[v][content] = 1  
                            
                    p = 3
                    if self.popularity_index[v][content] > p:
                        if self.view.has_cache(u):
                            copied = False
                        
                    break
                

                # No cache hits, get content from source
               
            self.controller.get_content(v)
            serving_node = v

        # Return content
        if v == source:
            copied = False
        #copied = False
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if not copied and v != receiver and self.view.has_cache(v):
                self.controller.put_content(v)
                copied = True
        self.controller.end_session()
        
        
@register_strategy('SCR10')
class Source_route_cache10(Strategy):
    """Source Cache strategy.

    implement local popularity and cooperation
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Source_route_cache10, self).__init__(view, controller)
        self.ridmap = []
        self.popularity_index = {}
        for i in self.view.topology().nodes():
            self.ridmap.append({})
            self.popularity_index[i] = {}                

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        res = {}
        for i in self.view.topology().nodes():
            if self.view.cache_lookup(i, content):
                res[i] = len(self.view.shortest_path(receiver, i))
                pathlen = res[i]
        for i in res:
            if res[i] < pathlen:
                pathlen = res[i]
                source = i
        
        
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        copied = True
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                        
                    if(content in self.popularity_index[v]):
                        self.popularity_index[v][content] = self.popularity_index[v][content] + 1
                    else:
                        self.popularity_index[v][content] = 1  
                            
                    p = 10
                    if self.popularity_index[v][content] > p:
                        if self.view.has_cache(u):
                            copied = False
                        
                    break
                # No cache hits, get content from source
               
            self.controller.get_content(v)
            serving_node = v

        # Return content
        if v == source:
            copied = False
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if not copied and v != receiver and self.view.has_cache(v):
                self.controller.put_content(v)
                copied = True
        self.controller.end_session()
        
@register_strategy('SCP3')
class Source_route_cacheP3(Strategy):
    """Source Cache strategy.

    implement local popularity and cooperation
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Source_route_cacheP3, self).__init__(view, controller)
        self.ridmap = []
        self.popularity_index = {}
        for i in self.view.topology().nodes():
            self.ridmap.append({})
            self.popularity_index[i] = {}                

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        res = {}
        for i in self.view.topology().nodes():
            if self.view.cache_lookup(i, content):
                res[i] = len(self.view.shortest_path(receiver, i))
                pathlen = res[i]
        for i in res:
            if res[i] < pathlen:
                pathlen = res[i]
                source = i
        
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        mark = 0
        cacheNode = []
        cooperation = False
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            
            count = 0
            if self.view.has_cache(v):

                if v != source:
                    if not self.view.is_cache_full(v):
                        cacheNode.append(v)
                    
                    if self.controller.get_content(v):
                        serving_node = v
                        
                        if(content in self.popularity_index[v]):
                            self.popularity_index[v][content] = self.popularity_index[v][content] + 1
                        else:
                            self.popularity_index[v][content] = 1  
                        
                        if content in self.ridmap[v]:
                            self.ridmap[v][content].append((receiver, time))
                        else:
                            self.ridmap[v][content] = [(receiver, time)]
                        
                        for i in self.ridmap[v][content]:
                            if time - i[1] > 1:
                                self.ridmap[v][content].remove(i)
                                
                        count = len(self.ridmap[v][content])
                        p = 3
                        if self.popularity_index[v][content] > p:
                            if self.view.has_cache(u):
                                cacheNode.append(u)

                        
                        break
                
                else:
                    if self.view.has_cache(u):
                        cacheNode.append(u)

                # No cache hits, get content from source
               
            self.controller.get_content(v)
            serving_node = v

        # Return content
        
        copied = False
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if not copied and v != receiver and self.view.has_cache(v):
                self.controller.put_content(v)
                copied = True
        self.controller.end_session()
        
        
        
        
