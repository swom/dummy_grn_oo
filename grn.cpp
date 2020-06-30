#include "grn.h"
#include <cassert>
#include <algorithm>

grn::grn(std::vector<int> nodes_per_layer,
         std::minstd_rand &rng ,
         double min_value,
         double max_value)
{
    for(size_t i = 0; i != nodes_per_layer.size(); i++)
    {
        std::vector<node> layer(nodes_per_layer[i]);
        if( i == 0)
        {
            m_layers.push_back(layer);
        }
        else
        {
            std::uniform_real_distribution<double> dist {min_value, max_value};
            std::vector< node *> prev_layer_nodes;

            for(auto& sending_node : m_layers[i - 1])
            {
                prev_layer_nodes.push_back(&sending_node);
            }
            for(auto& node : layer)
            {
                node.set_sending_nodes(prev_layer_nodes);

                std::vector<double> weights(prev_layer_nodes.size());
                std::generate(begin(weights), end(weights), [&dist, &rng](){
                    return dist(rng);
                });
                node.set_weights(weights);
                node.set_bias(dist(rng));

            }
            m_layers.push_back(layer);
        }
    }
}

bool operator==(const grn& lhs, const grn& rhs)
{
    return
            lhs.get_layers() == rhs.get_layers();
}
bool operator!=(const grn& lhs, const grn& rhs)
{
    return !(lhs == rhs);
}

double calc_mean_weights_biases(const grn& g)
{
    return calc_sum_weights_biases(g)/count_weights_biases(g);
}

double calc_sum_weights_biases(const grn& g)
{
    double sum = 0.0;
    for(const auto& layer : g.get_layers())
    {
        if(layer == g.get_layers()[0])
        {
            continue;
        }
        for(const auto& node : layer)
        {
            const auto& connections = node.get_connections_weights();
            sum += std::accumulate(connections.begin(), connections.end(), 0.0) + node.get_bias();
        }
    }
    return sum;
}

double calc_var_weights_biases(const grn& g)
{
    double var = 0.0;
    double mean = calc_mean_weights_biases(g);
    for(const auto& layer : g.get_layers())
    {
        if(layer == g.get_layers()[0])
        {
            continue;
        }
        for(const auto& node : layer)
        {
            var += (node.get_bias() - mean) * (node.get_bias() - mean);
            for(const auto& weight : node.get_connections_weights())
                var += (weight - mean) * (weight - mean);
        }
    }
    return var /= count_weights_biases(g);
}

int count_weights_biases(const grn& g)
{
    int n_weights_and_biases = 0;
    for(const auto& layer : g.get_layers())
    {
        if(layer == g.get_layers()[0])
        {
            continue;
        }
        n_weights_and_biases += static_cast<int>(layer.size());
        n_weights_and_biases +=
                std::accumulate(
                    layer.begin(),
                    layer.end(),
                    0,
                    [](int sum, const node& n)
        {return sum +  static_cast<int>(n.get_connections_weights().size());});
    }
    return n_weights_and_biases;
}

bool is_self_connected(const std::vector<node>& layer)
{
    for(const auto& focus_node : layer)
    {
        auto* ptr_node = &focus_node;
        for(const auto& comparison_node : layer)
        {
            auto sending_nodes = comparison_node.get_sending_nodes();
            if(std::find(sending_nodes.begin(),
                         sending_nodes.end(),
                         ptr_node) == sending_nodes.end())
                return false;
        }
    }
    return true;
}

grn mutate(const grn& g, std::minstd_rand& rng, double m_p, double m_s)
{
    grn mutated_grn = g;
    std::bernoulli_distribution p_mut(m_p);
    std::normal_distribution<double> s_mut(0.0, m_s);
    for(size_t i = 0; i != g.get_layers().size(); i++)
        for(size_t j = 0; j != g.get_layers()[i].size(); j++)
        {
            auto& old_node = g.get_layers()[i][j];
            auto& new_node = mutated_grn.access_layers()[i][j];
            if(p_mut(rng))
            {
                new_node.set_bias(old_node.get_bias() + s_mut(rng));
            }
            std::vector<double> new_weights = old_node.get_connections_weights();
            for(auto& connection : new_weights)
            {
                if(p_mut(rng))
                    connection += s_mut(rng);
            }
            new_node.set_weights(new_weights);
        }
    return mutated_grn;
}
//Horrible function
std::vector<node> self_connect(const std::vector<node>& layer)
{
    auto self_connected_layer = layer;
    //Create new pointers and new weight vectors of the layer itself
    std::vector<node *> nodes_pointers;
    std::vector<double> self_connections_weights;
    for(auto& node : self_connected_layer)
    {
        auto * ptr_node = &node;
        nodes_pointers.push_back(ptr_node);
        self_connections_weights.push_back(0);
    }

    //Append the new vectors to the already existing vectors of each node
    for(size_t i = 0; i != layer.size(); i++)
    {
        auto& node = self_connected_layer[i];
        node.get_sending_nodes().insert(
                    node.get_sending_nodes().end(),
                    nodes_pointers.begin(),
                    nodes_pointers.end() );

        node.get_connections_weights().insert(
                    node.get_connections_weights().end(),
                    self_connections_weights.begin(),
                    self_connections_weights.end());
    }

    return self_connected_layer;
}

grn take_input(const grn& g, std::vector<double> input)
{
    grn updated_input_grn = g;
    assert(g.get_layers()[0].size() == input.size());
    for(size_t i  = 0; i != input.size(); i++)
    {
        updated_input_grn.access_layers()[0][i].set_state(input[i]);
    }
    return updated_input_grn;
}

grn update_grn(const grn& g)
{
    grn updated_grn = g;
    for(size_t i = 0; i != g.get_layers().size(); i++)
    {
        updated_grn.access_layers()[i] = update_layer(g.get_layers()[i]);
    }
    return updated_grn;
}

std::vector<node> update_layer(const std::vector<node>& layer)
{
    std::vector<node> upd_layer = layer;
    for(size_t i = 0 ; i != layer.size(); i++)
    {
        upd_layer[i].set_state(receive_signal(layer[i]) > layer[i].get_bias());
    }
    return upd_layer;
}

void test_grn() noexcept
{

    ///A grn can be intizialized with a vector describing the numbers of node for each layer
    {
        // First create an instance of an engine.
        std::random_device rnd_device;
        std::minstd_rand rng {rnd_device()};

        std::vector<int> topology{1,2,3};
        grn g{topology, rng};
        for( size_t i = 0; i != g.get_layers().size(); i++)
        {
            assert(g.get_layers()[i].size() == static_cast<unsigned int>(topology[i]));
        }
    }

    ///Two grn can be checked for being the same or not
    {
        std::random_device rnd_device;
        std::minstd_rand rng {rnd_device()};
        std::vector<int> topology_1{1,2,3,4};
        std::vector<int> topology_2{1,2,3,4};

        grn g1{topology_1,rng};
        grn g2 = g1;
        assert(g1 == g2);
        //Change state of 1 node;
        g2.access_layers()[0][0].set_state(3.142356987);
        assert(g1 != g2);
        //change topology
        g2 = grn{topology_2,rng};
        assert(g1 != g2);

    }

    ///All nodes of a grn except input are connected to all the nodes of the previous layer
    {
        // First create an instance of an engine.
        std::random_device rnd_device;
        std::minstd_rand rng {rnd_device()};

        std::vector<int> topology{1,2,3,4,5,6,7};
        grn g{topology, rng};
        for( size_t i = 0; i != g.get_layers().size(); i++)
        {
            if( i == 0)
            {
                continue;
            }
            else
            {
                auto prev_layer = g.get_layers()[i - 1];
                auto layer = g.get_layers()[i];
                for(size_t j = 0; j != layer.size(); j++)
                {
                    auto sending_nodes = layer[j].get_sending_nodes();
                    assert(sending_nodes.size() == prev_layer.size());
                    for(size_t k = 0; k != sending_nodes.size(); k++)
                    {
                        assert(*sending_nodes[k] == prev_layer[k]);
                    }
                }
            }

        }

    }
    ///All weights and biases are intialized from a random distribution of a given range
    {
        // First create an instance of an engine.
        std::random_device rnd_device;
        std::minstd_rand rng {rnd_device()};
        double min_value = 0.4;
        double max_value = 0.6;
        double mean = 0;
        int tot_n_weights_and_biases = 0;
        int stop = 1000;
        int iterator = 0;

        std::vector<int> topology{1,1,2,2};
        while(iterator < stop)
        {
            grn g{topology,rng,min_value,max_value};
            for(const auto& layer : g.get_layers())
            {
                if(layer == g.get_layers()[0])
                {
                    continue;
                }
                else
                {
                    for(const auto& nodes : layer)
                    {
                        mean += nodes.get_bias();
                        tot_n_weights_and_biases++;

                        for(const auto& weight : nodes.get_connections_weights())
                        {
                            mean += weight;
                            tot_n_weights_and_biases++;
                        }
                    }
                }
            }

            iterator++;
        }

        mean /= tot_n_weights_and_biases;
        auto expected_mean = (max_value + min_value) / 2;
        assert(expected_mean - mean < 0.01
               && expected_mean - mean > -0.01);

    }

    //A layer can be updated
    {
        //Create an updater node that will update the layer
        node n;
        n.set_state(1);
        n.set_bias(0.5);

        //Create a layer
        std::vector<node> layer(2);

        //Create the vectors of connection and of pointers necessary to link
        // the nodes in the layer to the updater node
        auto weight_value = 1.0;
        std::vector<node *> sending_nodes{&n};
        std::vector<double> weight{weight_value};

        for(auto& nodes : layer)
        {
            nodes.set_state(0);
            nodes.set_bias(0.5);
            nodes.set_sending_nodes(sending_nodes);
            nodes.set_weights(weight);
        }

        auto updated_layer = update_layer(layer);
        assert(updated_layer != layer);
        for(const auto& node : updated_layer)
        {
            assert(node.get_state() == n.get_state() * weight_value);
        }
    }

    //A grn can take input
    {
        // First create an instance of an engine.
        std::random_device rnd_device;
        std::minstd_rand rng {rnd_device()};
        //Then create the grn
        std::vector<int> topology{2,1};
        grn g{topology,rng};

        std::vector<double> input{2,2};

        assert(g != take_input(g,input));
    }

    //A grn can be updated
    {
        // First create an instance of an engine.
        std::random_device rnd_device;
        std::minstd_rand rng {rnd_device()};
        //Then create the grn
        std::vector<int> topology{2,1};
        grn grn{topology,rng};
        //Set an input that will cause the grn to be updated
        std::vector<double> input{2,2};
        auto inputted_grn = take_input(grn,input);
        auto updated_grn = update_grn(inputted_grn);
        assert(inputted_grn != updated_grn);
    }

    //Horrible test
    //Some layers of the grn can be set to be self-connected
    {
        // First create an instance of an engine.
        std::random_device rnd_device;
        std::minstd_rand rng {rnd_device()};
        //Then create the grn
        std::vector<int> topology{2,2};
        grn grn{topology,rng};

        auto non_self_connected_layer = grn.get_layers()[1];
        assert(!(is_self_connected(non_self_connected_layer)));

        auto self_connected_layer = self_connect(grn.get_layers()[1]);
        assert(is_self_connected(self_connected_layer));

    }

    ///The mean weights of a grn can be calculated
    {
        // First create an instance of an engine.
        std::random_device rnd_device;
        std::minstd_rand rng {rnd_device()};
        //Then create the grn
        double weights_biases_value = 0.1;
        std::vector<int> topology{2,2};
        grn grn{topology,rng,weights_biases_value,weights_biases_value};

        assert(calc_mean_weights_biases(grn) - weights_biases_value < 0.01
               && calc_sum_weights_biases(grn) - weights_biases_value > -0.01);
    }
    //A grn can be mutated
    {
        // First create an instance of an engine.
        std::random_device rnd_device;
        std::minstd_rand rng {rnd_device()};
        //Then create the grn
        double weights_biases_value = 0.1;
        std::vector<int> topology{2,2};
        grn grn{topology,rng,weights_biases_value,weights_biases_value};

        double mut_step = 0.1;
        double mut_prob = 0.5;

        int bootstrap = 1000;
        int stop = 1000;
        double weight_bias_mean = 0.0;
        int n_weights_biases = 0;

        for( int i = 0 ; i!= stop; i++)
        {
            for(int j = 0; j != bootstrap; j++)
            {
                mutate(grn,rng,mut_step,mut_prob);
                weight_bias_mean += calc_sum_weights_biases(grn);
                n_weights_biases += count_weights_biases(grn);
            }
            weight_bias_mean /= n_weights_biases;
            assert(weight_bias_mean - weights_biases_value < 0.01
                   && weight_bias_mean - weights_biases_value > -0.01);
            weight_bias_mean = 0;
            n_weights_biases = 0;
        }
    }

    //A grn can be saved and loaded from a given file name
    {
        // First create an instance of an engine.
        std::random_device rnd_device;
        std::minstd_rand rng {rnd_device()};
        //Then create the grn
        double weights_biases_value = 0.1;
        std::vector<int> topology{2,2,2,5};
        grn grn{topology,rng,weights_biases_value,weights_biases_value};

    }


}
