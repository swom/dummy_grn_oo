#include "layer.h"
#include<cassert>

layer::layer(int n_nodes):
    m_nodes{static_cast<size_t>(n_nodes)}
{

}

void test_layer() noexcept
{
    ///A layer can be initialized with a given number of nodes
    {
        int number_of_nodes = 3;
        layer l{number_of_nodes};
        assert(l.get_nodes().size() == static_cast<unsigned int>(number_of_nodes));

    }
}
