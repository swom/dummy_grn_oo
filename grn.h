#ifndef GRN_H
#define GRN_H
#include "layer.h"
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
    const std::vector<layer>& get_layers() const noexcept {return m_layers;}

    ///Returns reference to rng
    std::minstd_rand& get_rng() noexcept {return  m_rng;}

private:

    ///The vector containing all the layers of the network
    std::vector<layer> m_layers;

    ///Reference to an random number generator (in simulation is going to belong to other classes)
    std::minstd_rand& m_rng;

};

void test_grn() noexcept;
#endif // GRN_H
