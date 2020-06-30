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

private:

    ///The vector containing all the layers of the network
    std::vector<std::vector<node>> m_layers;

};
///Controls that 2 grn are the same or not
bool operator==(const grn& lhs, const grn& rhs);
bool operator!=(const grn& lhs, const grn& rhs);

///Calculates the mean value of all weights and biases
/// EXCEPT INPUT/FIRST LAYER!
double calc_mean_weights_biases(const grn& g);

///Calculates the sum value of all weights and biases
/// EXCEPT INPUT/FIRST LAYER!
double calc_sum_weights_biases(const grn& g);

///Calculates the variance of all weights and biases
/// EXCEPT INPUT/FIRST LAYER!
double calc_var_weights_biases(const grn& g);

///Counts the total amount of weights + biases
/// in the network
/// EXCEPT INPUT/FIRST LAYER!
int count_weights_biases(const grn& g);

///Checks if a layer is self_connected
bool is_self_connected(const std::vector<node>& layer);

///Mutates a grn biases and weights
/// given a mutation probability and a mutation step
grn mutate(const grn& g, std::minstd_rand& rng, double m_p, double m_s);

///Fully connects the nodes of a layer to themeselves
std::vector<node> self_connect(const std::vector<node>& layer);

///Sets input layer(m_layers[0]) states to
/// the values given by an input vector
grn take_input(const grn&, std::vector<double> input);

///Updates the states of all nodes in a layer
/// based on the dot product between their connection weights
/// and the state of the nodes connected
std::vector<node> update_layer(const std::vector<node>& layer);

///Updates all layers of a grn
grn update_grn(const grn& g);

void test_grn() noexcept;
#endif // GRN_H
