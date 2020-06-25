#include "grn.h"
#include <cassert>
#include <algorithm>

grn::grn(std::vector<int> nodes_per_layer,
         std::minstd_rand &rng ,
         double min_value,
         double max_value):
    m_rng{rng}
{
    for(size_t i = 0; i != nodes_per_layer.size(); i++)
    {
        layer l(nodes_per_layer[i]);
        if( i == 0)
        {
            m_layers.push_back(l);
        }
        else
        {
            std::uniform_real_distribution<double> dist {min_value, max_value};
            std::vector< node *> prev_layer_nodes;

            for(auto& sending_node : m_layers[i - 1].get_nodes())
            {
                prev_layer_nodes.push_back(&sending_node);
            }
            for(auto& node : l.get_nodes())
            {
                node.set_sending_nodes(prev_layer_nodes);

                std::vector<double> weights(prev_layer_nodes.size());
                std::generate(begin(weights), end(weights), [&dist, &rng](){
                    return dist(rng);
                });
                node.set_weights(weights);

            }
            m_layers.push_back(l);

        }
    }
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
            assert(g.get_layers()[i].get_nodes().size() == static_cast<unsigned int>(topology[i]));
        }
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
                for(size_t j = 0; j != layer.get_nodes().size(); j++)
                {
                    auto sending_nodes = layer.get_nodes()[j].get_sending_nodes();
                    assert(sending_nodes.size() == prev_layer.get_nodes().size());
                    for(size_t k = 0; k != sending_nodes.size(); k++)
                    {
                        assert(*sending_nodes[k] == prev_layer.get_nodes()[k]);
                    }
                }
            }

        }

    }
    ///All weights are intialized from a random distribution of a given range
    {
        // First create an instance of an engine.
        std::random_device rnd_device;
        std::minstd_rand rng {rnd_device()};
        double min_value = 0.4;
        double max_value = 0.6;
        double mean = 0;
        int tot_n_weights = 0;
        int stop = 1000;
        int iterator = 0;

        std::vector<int> topology{1,1,2,2};
        while(iterator < stop)
        {
            grn g{topology,rng,min_value,max_value};
            for(const auto& layer : g.get_layers())
                for(const auto& nodes : layer.get_nodes())
                    for(const auto& weight : nodes.get_connections_weights())
                    {
                        mean += weight;
                        tot_n_weights++;
                    }
            iterator++;
        }

        mean /= tot_n_weights;
        auto expected_mean = (max_value + min_value) / 2;
        assert(expected_mean - mean < 0.01
               && expected_mean - mean > -0.01);

    }


}
