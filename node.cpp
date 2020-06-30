#include "node.h"
#include<cassert>
#include<numeric>
#include <functional>

node::node(double bias,
           double state):
    m_bias{bias},
    m_state{state}
{
    assert(m_incoming_nodes.empty());
    assert(m_incoming_weights.empty());
    assert(m_bias - bias < 0.0001 &&m_bias - bias > -0.0001);
    assert(m_state - state < 0.00001 && m_state - state > -0.0001);
}

bool operator== (const node& lhs, const node& rhs)
{
    return lhs.get_bias() == rhs.get_bias()
            && lhs.get_state() == rhs.get_state()
            && lhs.get_sending_nodes() == rhs.get_sending_nodes()
            && lhs.get_connections_weights() == rhs.get_connections_weights();
}

bool operator!= (const node& lhs, const node& rhs)
{
    return !(lhs == rhs);
}

double receive_signal(const node& n) noexcept
{
    return std::inner_product(n.get_connections_weights().begin(),
                              n.get_connections_weights().end(),
                              n.get_sending_nodes().begin(),
                              0.0,
                              std::plus<double>(),
                              [](double a, node* b){ return a * (b->get_state() > b->get_bias());});
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
        node n;
        assert(n.get_connections_weights().empty());
        assert(n.get_sending_nodes().empty());
        assert(n.get_bias() < 0.0001 && n.get_bias() > -0.0001);
        assert(n.get_state() < 0.00001 && n.get_state() > -0.0001);
    }

    //A node receives a signal equal to the sum of the
    //products of connection weight
    //belonging to nodes whose state is above the bias
    {
        std::vector<node> sending_nodes(2);
        auto node_value = 0.1;
        auto node_bias = 0.05;
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
        auto received_signal = receive_signal(n);
        auto expected_signal = (node_value > node_bias) * connection_weight * sending_nodes.size();
        auto signal_delta = received_signal - expected_signal;
                assert(signal_delta < 0.0001
                       && signal_delta > -0.0001);
    }

}
