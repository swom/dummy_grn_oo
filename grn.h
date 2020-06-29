#ifndef GRN_H
#define GRN_H
#include "node.h"
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>


class grn
{
public:

    grn(std::vector<int> nodes_per_layer,
        std::minstd_rand& rng,
        double min_value =  0.5,
        double max_value = 0.5);

    ///Returns const ref to layers
    const std::vector<std::vector<node>>& get_layers() const noexcept {return m_layers;}

    ///Returns  ref to layers
    std::vector<std::vector<node>>& access_layers() noexcept {return m_layers;}

    ///Returns reference to rng
    std::minstd_rand& get_rng() noexcept {return  m_rng;}

private:

    ///The vector containing all the layers of the network
    std::vector<std::vector<node>> m_layers;

    ///Reference to an random number generator (in simulation is going to belong to other classes)
    std::minstd_rand& m_rng;

};
///Controls that 2 grn are the same or not
bool operator==(const grn& lhs, const grn& rhs);
bool operator!=(const grn& lhs, const grn& rhs);

///Updates the states of all nodes in a layer
/// based on the dot product between their connection weights
/// and the state of the nodes connected
std::vector<node> update_layer(const std::vector<node>& layer);

///Updates all layers of a grn
grn update_grn(const grn& g);

void test_grn() noexcept;
#endif // GRN_H
