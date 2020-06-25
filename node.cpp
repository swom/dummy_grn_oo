#include "node.h"
#include<cassert>
#include<numeric>
#include <functional>

node::node()
{

}

bool operator== (const node& lhs, const node& rhs)
{
    return lhs.get_bias() == rhs.get_bias()
            && lhs.get_state() == rhs.get_state()
            && lhs.get_sending_nodes() == rhs.get_sending_nodes()
            && lhs.get_connections_weights() == rhs.get_connections_weights();
}

double receive_signal(const node& n) noexcept
{
    return std::inner_product(n.get_connections_weights().begin(),
                              n.get_connections_weights().end(),
                              n.get_sending_nodes().begin(),
                              0,
                              [](double a, double b){return a + b;},
    [](double a, node* b){ return a * b->get_state();});
}

void test_node() noexcept
{
    ///A node is initialized with
    /// a vector of connections weights from other nodes
    /// a vector of pointers to sending a signal to this node
    /// a bias value
    /// a state value
    /// by default all is 0 or empty
    {
        node n{};
        assert(n.get_connections_weights().empty());
        assert(n.get_sending_nodes().empty());
        assert(n.get_bias() < 0.0001 && n.get_bias() > -0.0001);
        assert(n.get_state() < 0.00001 && n.get_state() > -0.0001);
    }

    //A node receives a signal equal to the sum of the
    //products of connection wieght and sending node value
    {
        std::vector<node> sending_nodes(2);
        auto node_value = 0.1;
        auto connection_weight = 0.1;

        std::vector<node *> ptrs_to_nodes;
        std::vector<double> connection_weights;
        for( auto& sending_node : sending_nodes)
        {
            sending_node.set_state(node_value);
            connection_weights.push_back(connection_weight);
            ptrs_to_nodes.push_back(&sending_node);
        }

        node n;
        n.set_weights(connection_weights);
        n.set_sending_nodes(ptrs_to_nodes);
        auto signal_delta = receive_signal(n) - (node_value * connection_weight * sending_nodes.size());
                assert(signal_delta < 0.0001
                       && signal_delta > -0.0001);
    }

}
