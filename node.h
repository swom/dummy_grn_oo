#ifndef NODE_H
#define NODE_H

#include"connection.h"
#include<vector>

class node
{
public:
    node();

    ///Returns the value of the bias
    double get_bias() const noexcept { return  m_bias;}

    ///Returns the value of the state of the network
    double get_state() const noexcept { return m_state;}

    ///Returns const ref to the vector of connections
    const std::vector<connection>& get_connections() const noexcept { return m_connections;}

private:
    ///The vector of connections from which
    ///  the node will RECEIVE signals
    std::vector<connection> m_connections;

    ///The bias acting on the node
    /// FOR NOW it behaves like a treshold value
    double m_bias;

    ///The value of the internal state of the node
    /// Which corresponds to the sum of the incoming
    /// signal
    double m_state;
};

void test_node() noexcept;

#endif // NODE_H
