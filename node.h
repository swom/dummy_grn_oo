#ifndef NODE_H
#define NODE_H

#include<vector>
#include"connection.h"

class node
{
public:
    node();

    ///Returns the value of the bias
    double get_bias() const noexcept { return  m_bias;}

    ///Returns the value of the state of the network
    double get_state() const noexcept { return m_state;}

    ///Returns const ref to the vector of connections weights
    const std::vector<double>& get_connections_weights() const noexcept { return m_incoming_weights;}

    ///Return const ref to vector of pointers to sending nodes
    const std::vector<node *>& get_sending_nodes() const noexcept {return m_incoming_nodes;}

    ///sets vector of pointers to sending nodes
    void set_sending_nodes(const std::vector<node *>& sending_nodes) noexcept { m_incoming_nodes = sending_nodes;}

    ///sets the state of the network
    void set_state(double state) noexcept {m_state = state;}

    ///sets vector of weights of sending nodes connections
    void set_weights(const std::vector<double>& weights) noexcept { m_incoming_weights = weights;}

private:
    ///The vector of weights of the connections from which
    ///  the node will RECEIVE signals
    std::vector<double> m_incoming_weights;

    ///The vector of pointers to the nodes from which
    ///  the node will RECEIVE signals
    std::vector<node *> m_incoming_nodes;

    ///The bias acting on the node
    /// FOR NOW it behaves like a treshold value
    double m_bias;

    ///The value of the internal state of the node
    /// Which corresponds to the sum of the incoming
    /// signal
    double m_state;
};

///Checks that two nodes are the same
bool operator==(const node& lhs, const node& rhs);


///Calculates the signal received by a node
double receive_signal(const node& n) noexcept;

void test_node() noexcept;

#endif // NODE_H
