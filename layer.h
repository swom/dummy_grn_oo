#ifndef LAYER_H
#define LAYER_H
#include "node.h"

#include<vector>

class layer
{
public:
    layer(int n_nodes = 0);

    ///Returns const ref to m_nodes
    const std::vector<node>& get_nodes() const noexcept {return m_nodes;}

private:
    std::vector<node> m_nodes;

};

void test_layer() noexcept;

#endif // LAYER_H
