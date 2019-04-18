import unittest

import fnss

from icarus.scenarios import IcnTopology
import icarus.models as strategy
from icarus.execution import NetworkModel, NetworkView, NetworkController, TestCollector


class MyTestCase(unittest.TestCase):
    
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass
        
    def my_topology(cls):
        """Return my topology for testing caching strategies
        """
        # Topology sketch
        #            0
        #         /     \
        #        /       \
        #       /         \
        #      1           2
        #    /   \       /  \
        #   3     4     5    6
        #  / \   / \   / \  / \
        # 7   8 9  10 11 1213 14
        #
        k = 2;
        h = 3;
        delay = 5;
        topology = IcnTopology(fnss.k_ary_tree_topology(k, h))
        receivers = [v for v in topology.nodes_iter()
                     if topology.node[v]['depth'] == h]
        sources = [v for v in topology.nodes_iter()
                   if topology.node[v]['depth'] == 0]
        routers = [v for v in topology.nodes_iter()
                  if topology.node[v]['depth'] > 0
                  and topology.node[v]['depth'] < h]
        topology.graph['icr_candidates'] = set(routers)
        for v in receivers:
            fnss.add_stack(topology, v, 'receiver')
        for v in routers:
            fnss.add_stack(topology, v, 'router', {'cache_size': 2})

           
        contents = (1, 2, 3)
        fnss.add_stack(topology, source, 'source', {'contents': contents})
            
        # set weights and delays on all links
        fnss.set_weights_constant(topology, 1.0)
        fnss.set_delays_constant(topology, delay, 'ms')
        fnss.set_delays_constant(topology, 20, 'ms', [(0,1),(0,2)])
        # label links as internal
        for u, v in topology.edges_iter():
            topology.edge[u][v]['type'] = 'internal'
        return IcnTopology(topology)

    def setUp(self):
        topology = self.my_topology()
        model = NetworkModel(topology, cache_policy={'name': 'LRU'})
        self.view = NetworkView(model)
        self.controller = NetworkController(model)
        self.collector = TestCollector(self.view)
        self.controller.attach_collector(self.collector)

    def tearDown(self):
        pass
    
    # ²âÊÔÓÃÀý
    def test_case1(self):
        hr = strategy.CRAN_CountBloomFilter1(self.view, self.controller)
        # receiver 7 requests 2, expect miss
        hr.process_event(1, 13, 1, True)
        loc = self.view.content_locations(2)
        self.assertEquals(len(loc), 2)
        self.assertIn(1, loc)
        self.assertIn(0, loc)
        
        summary = self.collector.session_summary()
        #exp_req_hops = [(7, 3), (3, 1), (1, 0)]
        #exp_cont_hops = [(0, 1), (1, 3), (3, 7)]
        req_hops = summary['request_hops']
        cont_hops = summary['content_hops']
        self.assertSetEqual(set(exp_req_hops), set(req_hops))
        self.assertSetEqual(set(exp_cont_hops), set(cont_hops))
        
        
        
        hr.process_event(1, 13, 2, True)
        
        for i in range(2, 8):
            print "loop", i
            hr.process_event(1, 16, i, True)

        num = self.view.cache_dump(4)
        print "cache of node 4:", num
        num2 = self.view.cache_dump(5)
        print "cache of node 5:", num2
        num1 = self.view.cache_dump(1)
        print "cache of node 1:", num1


        hr.process_event(1, 13, 6, True)
        hr.process_event(1, 13, 6, True)
        hr.process_event(1, 13, 6, True)
        hr.process_event(1, 13, 7, True)
        hr.process_event(1, 13, 7, True)
        hr.process_event(1, 13, 5, True)
        hr.process_event(1, 13, 5, True)
        hr.process_event(1, 13, 1, True)
        hr.process_event(1, 13, 8, True)
        num = self.view.cache_dump(4)
        print "cache of node 4:", num
        num2 = self.view.cache_dump(5)
        print "cache of node 5:", num2
        num1 = self.view.cache_dump(1)
        print "cache of node 1:", num1
         
        

if __name__ == '__main__':
    unittest.main()