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

grn update_grn(const grn& g)
{
    grn updated_grn = g;
    for(auto layer : g.get_layers())
    {
        layer = update_layer(layer);
    }
    assert(g != updated_grn);
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
        g2.access_layers()[1][2].set_state(3.142356987);
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
                for(const auto& nodes : layer)
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

    //A grn can be updated
    {
        // First create an instance of an engine.
        std::random_device rnd_device;
        std::minstd_rand rng {rnd_device()};
        //Then create the grn
        std::vector<int> topology{2,1};
        grn grn{topology,rng};
        auto updated_grn = update_grn(grn);
    }



}
