#include "node.h"
#include<cassert>

node::node()
{

}

void test_node() noexcept
{
    ///A node is initialized with
    ///  a list of connections coming from other nodes
    /// a bias value
    /// a state value
    /// by default all is 0 or empty
    {
        node n{};
        assert(n.get_connections().empty());
        assert(n.get_bias() < 0.0001 && n.get_bias() > -0.0001);
        assert(n.get_state() < 0.00001 && n.get_state() > -0.0001);
    }

}
