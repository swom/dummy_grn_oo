#include "grn.h"
#include <cassert>
#include <numeric>



GRN::GRN(size_t n_input, size_t n_hidden, size_t n_output, double weights):
    m_ConI2H(n_input,std::vector<double>(n_hidden, weights)),
    m_ConH2H(n_hidden,std::vector<double>(n_hidden, weights)),
    m_ConH2O(n_hidden,std::vector<double>(n_output, weights)),
    m_THidden(n_hidden,0),
    m_TOutput(n_output,0),
    m_ExInput(n_input,0),
    m_ExHidden(n_hidden,true),
    m_ExOutput(n_output,true)
{

}

GRN::GRN(std::vector<std::vector<double> > ConI2H,
         std::vector<std::vector<double> > ConH2H,
         std::vector<std::vector<double> > ConH2O,
         std::vector<double> THidden,
         std::vector<double> TOutput,
         std::vector<bool>ExHidden,
         int n_input,
         int n_output):
    m_ConI2H{ ConI2H},
    m_ConH2H{ ConH2H},
    m_ConH2O{ ConH2O},
    m_THidden{ THidden},
    m_TOutput{ TOutput},
    m_ExInput(n_input, 0),
    m_ExHidden{ExHidden},
    m_ExOutput(n_output, 0)
{

}

GRN::GRN(std::vector<int> nodes_per_layer):
    m_layers(nodes_per_layer.size())
{
    for(size_t i = 0 ; i != m_layers.size(); i++)
    {
        m_layers[i] = layer{nodes_per_layer[i]};
    }
}

bool operator==(const GRN& lhs, const GRN& rhs)
{
    return
            lhs.get_H2H() == rhs.get_H2H()
            && lhs.get_H2O() == rhs.get_H2O()
            && lhs.get_I2H() == rhs.get_I2H()
            && lhs.get_hid_tresh() == rhs.get_hid_tresh()
            && lhs.get_out_tresh() == rhs.get_out_tresh()
            && lhs.get_hidden_nodes() == rhs.get_hidden_nodes();
    //We do not look at input and output nodes since they might vary
    //depending on the status of the simulation
    //this might be true also for hidden nodes,
    //but not before the first time step of a cycle
}

bool operator!=(const GRN& lhs, const GRN& rhs)
{
    return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os, const GRN& p)
{
    save_I2H_weights(os,p);
    save_H2H_weights(os,p);
    save_H2O_weights(os,p);
    save_THidden(os,p);
    save_TOutput(os,p);
    save_ExHidden(os,p);
    save_n_input_nodes(os, p);
    save_n_output_nodes(os, p);

    return os;
}

std::ifstream& operator>>(std::ifstream& is, GRN& grn)
{
    auto ConI2H = load_I2H_weights( is);
    auto ConH2H = load_H2H_weights( is);
    auto ConH2O = load_H2O_weights( is);
    auto THidden = load_hid_tresh( is);
    auto TOutput = load_out_tresh( is);
    auto ExHidden = load_hid_state( is);
    int n_input = load_n_input_nodes( is);
    int n_output = load_n_input_nodes( is);

    grn = GRN{ConI2H,
            ConH2H,
            ConH2O,
            THidden,
            TOutput,
            ExHidden,
            n_input,
            n_output};

    return is;
}

void GRN::set_hid_node(int index_node, bool state)
{m_ExHidden[static_cast<unsigned int>(index_node)] = state;}

void GRN::set_all_H2H(double value) noexcept
{
    for (auto& node : get_H2H())
    {
        for(auto& connection_weight : node)
            connection_weight = value;
    }
}

void GRN::set_all_H2O(double value) noexcept
{
    for (auto& node : get_H2O())
    {
        for(auto& connection_weight : node)
            connection_weight = value;
    }
}

void GRN::set_all_I2H(double value) noexcept
{
    for (auto& node : get_I2H())
    {
        for(auto& connection_weight : node)
            connection_weight = value;
    }
}

void GRN::set_all_out_nodes(bool state)
{
    for(int i = 0; i != static_cast<int>(get_output_nodes().size()); i++)
        set_out_node(i, state);
}


void GRN::set_all_hid_nodes(bool state)
{
    for(int i = 0; i != static_cast<int>(get_hidden_nodes().size()); i++)
        set_hid_node(i,state);
}

void GRN::set_all_hid_tresh(double value)
{
    for(auto& treshold : get_hid_tresh())
        treshold = value;
}

void GRN::set_all_out_tresh(double value)
{
    for(auto& treshold : get_out_tresh())
        treshold = value;
}

void GRN::set_inputs(std::vector<double> inputs)
{
    assert(inputs.size() == get_input_nodes().size());
    get_input_nodes().swap(inputs);
}

void GRN::set_out_node(int index_node, bool state)
{m_ExOutput[static_cast<unsigned int>(index_node)] = state;}

GRN load_grn( const std::string& filename)
{
    GRN g;
    std::ifstream  i_f(filename);
    i_f >> g;
    return g;
}

std::vector<std::vector<double>> load_I2H_weights(std::ifstream& is)
{
    std::string dummy; // To remove the annotation in the file
    std::string line;
    std::string subline;
    std::vector<std::vector<double> > ConI2H;

    //Get everything until character that signals the end of the layer
    std::getline(is, line, '|');

    std::stringstream iss(line);
    while (std::getline(iss, subline ,'!').good())
    {
        std::stringstream isss(subline);
        std::vector<double> node_weights;
        double tmp;
        while(isss >> tmp)
        {
            node_weights.push_back(tmp);
            isss >> dummy;
        }
        ConI2H.push_back(node_weights);
        iss >> dummy; //Eat the " , " after the " ! "
    }
    is >> dummy; //Eat the " , " after the " | "

    return ConI2H;
}

std::vector<std::vector<double>> load_H2H_weights(std::ifstream& is)
{
    std::string dummy; // To remove the annotation in the file
    std::string line;
    std::string subline;
    std::vector<std::vector<double> > ConH2H;

    //Get everything until character that signals the end of the layer
    std::getline(is, line, '|');

    std::stringstream iss(line);
    while (std::getline(iss, subline ,'!').good())
    {
        std::stringstream isss(subline);
        std::vector<double> node_weights;
        double tmp;
        while(isss >> tmp)
        {
            node_weights.push_back(tmp);
            isss >> dummy;
        }
        ConH2H.push_back(node_weights);
        iss >> dummy; //Eat the " , " after the " ! "
    }
    is >> dummy; //Eat the " , " after the " | "

    return ConH2H;
}

std::vector<std::vector<double>> load_H2O_weights(std::ifstream& is)
{
    std::string dummy; // To remove the annotation in the file
    std::string line;
    std::string subline;
    std::vector<std::vector<double> > ConH2O;

    //Get everything until character that signals the end of the layer
    std::getline(is, line, '|');

    std::stringstream iss(line);
    while (std::getline(iss, subline ,'!').good())
    {
        std::stringstream isss(subline);
        std::vector<double> node_weights;
        double tmp;
        while(isss >> tmp)
        {
            node_weights.push_back(tmp);
            isss >> dummy;
        }
        ConH2O.push_back(node_weights);
        iss >> dummy; //Eat the " , " after the " ! "
    }
    is >> dummy; //Eat the " , " after the " | "

    return ConH2O;
}

std::vector<double> load_hid_tresh(std::ifstream& is)
{
    std::string dummy; // To remove the annotation in the file
    std::string line;
    std::vector<double> THidden;

    //Get everything until character that signals the end of the layer
    std::getline(is, line, '|');
    std::stringstream iss(line);
    double tmp;
    while(iss >> tmp)
    {
        THidden.push_back(tmp);
        iss >> dummy;
    }
    is >> dummy; //Eat the " , " after the " | "

    return THidden;
}

std::vector<bool> load_hid_state(std::ifstream& is)
{
    std::string dummy; // To remove the annotation in the file
    std::string line;
    std::vector<bool> ExHidden;

    //Get everything until character that signals the end of the layer
    std::getline(is, line, '|');
    std::stringstream iss(line);
    bool temporary;

    while(iss >> temporary)
    {
        ExHidden.push_back(temporary);
        iss >> dummy;
    }
    is >> dummy; //Eat the " , " after the " | "

    return ExHidden;
}


std::vector<double> load_out_tresh(std::ifstream& is)
{
    std::string dummy; // To remove the annotation in the file
    std::string line;
    std::vector<double> TOutput;

    //Get everything until character that signals the end of the layer
    std::getline(is, line, '|');
    std::stringstream iss(line);
    double tmp;
    while(iss >> tmp)
    {
        TOutput.push_back(tmp);
        iss >> dummy;
    }
    is >> dummy; //Eat the " , " after the " | "
    return TOutput;
}

int load_n_input_nodes(std::ifstream& is)
{
    std::string dummy; // To remove the annotation in the file
    std::string line;

    //Get everything until character that signals the end of the layer
    std::getline(is, line, '|');
    std::stringstream iss(line);
    int n_input_nodes;
    while(iss >> n_input_nodes)
    {
        iss >> dummy;
    }
    is >> dummy; //Eat the " , " after the " | "
    return n_input_nodes;
}

int load_n_output_nodes(std::ifstream& is)
{
    std::string dummy; // To remove the annotation in the file
    std::string line;

    //Get everything until character that signals the end of the layer
    std::getline(is, line, '|');
    std::stringstream iss(line);
    int n_output_nodes;
    while(iss >> n_output_nodes)
    {
        iss >> dummy;
    }
    is >> dummy; //Eat the " , " after the " | "
    return n_output_nodes;
}


std::vector<double> hid_updates_hid(GRN& g) noexcept
{
    std::vector<double> signal_from_hid;
    for(unsigned int i = 0; i != g.get_hidden_nodes().size(); i++)
    {
        double signal_hid = 0;
        for (unsigned int j = 0 ; j != g.get_hidden_nodes().size(); j++)
        {
            signal_hid += g.get_hidden_nodes()[i] * g.get_H2H()[j][i];
        }
        signal_from_hid.push_back(signal_hid);
    }
    return signal_from_hid;
}

void hid_updates_out(GRN& g) noexcept
{
    for(unsigned int i = 0; i != g.get_output_nodes().size(); i++)
    {
        double signal_out = 0;
        for (unsigned int j = 0 ; j != g.get_hidden_nodes().size(); j++)
        {
            signal_out  += g.get_hidden_nodes()[j] * g.get_H2O()[j][i];

        }
        if(signal_out > g.get_out_tresh()[i]) {g.set_out_node(static_cast<int>(i),true);}
        else {g.set_out_node(static_cast<int>(i),false);}
    }
}

std::vector<double> inp_updates_hid(GRN& g) noexcept
{
    std::vector<double> signal_from_input;
    for(unsigned int i = 0; i != g.get_hidden_nodes().size(); i++)
    {
        double signal_hid = 0;
        for (unsigned int j = 0 ; j != g.get_input_nodes().size(); j++)
        {
            signal_hid += g.get_input_nodes()[i] * g.get_I2H()[j][i];
        }
        signal_from_input.push_back(signal_hid);
    }
    return signal_from_input;
}

void jordi_response_mech(GRN& g) //Jordi style
{
    //First he updates output from hidden (is not the first thing he does
    //in the code but is the first thing that actually changes something in the
    //GRN
    hid_updates_out(g);
    //Update the state of the hidden nodes
    update_hid(g);
}


void mutation_I2H(GRN& g, std::minstd_rand& rng,
                  std::bernoulli_distribution& mu_p,
                  std::normal_distribution<double> mu_st) noexcept
{
    for(auto & node : g.get_I2H())
        for(auto & weight : node)
        {
            if(mu_p(rng)){weight += mu_st(rng);}
        }
}

void mutation_H2H(GRN& g, std::minstd_rand& rng,
                  std::bernoulli_distribution& mu_p,
                  std::normal_distribution<double> mu_st) noexcept
{
    for(auto & node : g.get_H2H())
        for(auto & weight : node)
        {
            if(mu_p(rng)){weight += mu_st(rng);}
        }
}

void mutation_H2O(GRN& g, std::minstd_rand& rng,
                  std::bernoulli_distribution& mu_p,
                  std::normal_distribution<double> mu_st) noexcept
{
    for(auto & node : g.get_H2O())
        for(auto & weight : node)
        {
            if(mu_p(rng)){weight += mu_st(rng);}
        }
}

void mutation_hid_tr(GRN& g, std::minstd_rand& rng,
                     std::bernoulli_distribution& mu_p,
                     std::normal_distribution<double> mu_st) noexcept
{
    for(auto & treshold : g.get_hid_tresh())
    {
        if(mu_p(rng)){treshold += mu_st(rng);}
    }
}

void mutation_out_tr(GRN& g, std::minstd_rand& rng,
                     std::bernoulli_distribution& mu_p,
                     std::normal_distribution<double> mu_st) noexcept
{
    for(auto & treshold : g.get_out_tresh())
    {
        if(mu_p(rng)){treshold += mu_st(rng);}
    }
}

void mutation(GRN& g, std::minstd_rand& rng,
              std::bernoulli_distribution& mu_p,
              std::normal_distribution<double> mu_st) noexcept
{
    mutation_I2H(g,rng,mu_p,mu_st);
    mutation_H2H(g,rng,mu_p,mu_st);
    mutation_H2O(g,rng,mu_p,mu_st);
    mutation_hid_tr(g,rng,mu_p,mu_st);
    mutation_out_tr(g,rng,mu_p,mu_st);
}

int n_connections(const GRN& g) noexcept
{
    return static_cast<int> (
                g.get_H2H().size() * g.get_H2H()[0].size() +
            g.get_H2O().size() * g.get_H2O()[0].size() +
            g.get_I2H().size() * g.get_I2H()[0].size());
}

void save_grn(const GRN& grn, const std::string& filename)
{
    std::ofstream f(filename);
    f << grn;
}

std::ostream& save_I2H_weights(std::ostream& os, const GRN& p)
{
    for(const auto& node : p.get_I2H())
    {
        for(const auto& weight : node)
        {
            os << weight << " , ";
        }
        os << " ! , ";
    }
    os << " | , ";
    return os;
}

std::ostream& save_H2H_weights(std::ostream& os, const GRN& p)
{
    for(const auto& node : p.get_H2H())
    {
        for(const auto& weight : node)
        {
            os << weight << " , ";
        }
        os << " ! , ";
    }
    os << " | , ";
    return os;
}

std::ostream& save_H2O_weights(std::ostream& os, const GRN& p)
{
    for(const auto& node : p.get_H2O())
    {
        for(const auto& weight : node)
        {
            os << weight << " , ";
        }
        os << " ! , ";
    }
    os << " | , ";
    return os;
}

std::ostream& save_THidden(std::ostream& os, const GRN& p)
{
    for(const auto& treshold : p.get_hid_tresh())
    {
        os << treshold << " , ";
    }
    os << " | , ";
    return os;
}

std::ostream& save_TOutput(std::ostream& os, const GRN& p)
{
    for(const auto& treshold : p.get_out_tresh())
    {
        os << treshold << " , ";
    }
    os << " | , ";
    return os;
}

std::ostream& save_ExHidden(std::ostream& os, const GRN& p)
{
    int state;
    for(const auto& boolstate : p.get_hidden_nodes())
    {
        state = boolstate;
        os << state << " , ";
    }
    os << " | , ";
    return os;
}

std::ostream& save_n_input_nodes(std::ostream& os, const GRN& grn)
{
    int n_inputs = grn.get_input_nodes().size();
    os << n_inputs << " , "
       << " | " << " , ";
    return  os;
}

std::ostream& save_n_output_nodes(std::ostream& os, const GRN& grn)
{
    int n_outputs = grn.get_output_nodes().size();
    os << n_outputs << " , "
       << " | " << " , ";
    return os;
}

double sum_I2H(const GRN& g) noexcept
{
    double sum_I2H = 0;
    for (const auto& node : g.get_I2H()) {
        sum_I2H += std::accumulate(node.begin(), node.end(), 0.0);
    }
    return sum_I2H;
}

double sum_H2H(const GRN& g) noexcept
{
    double sum_H2H = 0;
    for (const auto& node : g.get_H2H()) {
        sum_H2H += std::accumulate(node.begin(), node.end(), 0.0);
    }
    return sum_H2H;
}

double sum_H2O(const GRN& g) noexcept
{
    double sum_H2O = 0;
    for (const auto& node : g.get_H2O()) {
        sum_H2O += std::accumulate(node.begin(), node.end(), 0.0);
    }
    return sum_H2O;
}

void update_hid(GRN& g) noexcept
{
    auto signal_in = inp_updates_hid(g);
    auto signal_hid = hid_updates_hid(g);
    assert(signal_in.size() == signal_hid.size());
    for(size_t i = 0; i != signal_in.size(); i++)
    {
        if((signal_in[i] + signal_hid[i]) > g.get_hid_tresh()[i])
        {g.set_hid_node(static_cast<int>(i),true);}
        else
        {g.set_hid_node(static_cast<int>(i),false);}
    }
}

double weights_sum (const GRN& g) noexcept
{
    return sum_H2H(g) + sum_H2O(g) + sum_I2H(g);
}

double weights_mean (const GRN& g) noexcept
{
    return weights_sum(g)/n_connections(g);
}

double weights_var(const GRN& g) noexcept
{
    double var = 0;
    double mean = weights_mean(g);
    for(const auto& node : g.get_H2H())
        for(const auto& weight : node)
        {
            var += (weight - mean) * (weight - mean);
        }
    for(const auto& node : g.get_H2O())
        for(const auto& weight : node)
        {
            var += (weight - mean) * (weight - mean);
        }
    for(const auto& node : g.get_I2H())
        for(const auto& weight : node)
        {
            var += (weight - mean) * (weight - mean);
        }
    return var /= n_connections(g);
}


void test_GRN()//!OCLINT , tests may be long
{
#ifndef NDEBUG
    //A grn is initialized with connection weights
    //all to a given value, default  = 1
    {
        double weight_value = 3.14;
        GRN g{3,3,2,weight_value};
        for(const auto& node : g.get_I2H())
        {
            for(const auto& weight : node)
                assert(weight - weight_value < 0.00001 && weight - weight_value > -0.00001);
        }
        for(const auto& node : g.get_H2H())
        {
            for(const auto& weight : node)
                assert(weight - weight_value < 0.00001 && weight - weight_value > -0.00001);
        }
        for(const auto& node : g.get_H2O())
        {
            for(const auto& weight : node)
                assert(weight - weight_value < 0.00001 && weight - weight_value > -0.00001);
        }
    }
    //A GRN is initialized with three layers
    // and a certain amount of nodes in each layer
    //These hyperparameters cannot be altered
    //during the simulation therefore no set() functions.
    //By default 3-3-2
    {
        size_t n_input = 5;
        size_t n_hidden = 5;
        size_t n_output = 5;
        GRN g(n_input,n_hidden,n_output);
        assert(g.get_I2H().size() == n_input);
        assert(g.get_H2H().size() == n_hidden);
        assert(g.get_H2O().size() == n_output);
    }

    //It is possible to get the values of the nodes states
    {
        GRN g;
        for(auto input : g.get_input_nodes())
            assert(input < 0.00001 && input > -0.00001);
        for(auto hidden : g.get_hidden_nodes())
            assert(hidden);
        for(auto output : g.get_output_nodes())
            assert(output);
    }

    //All weights in H2O connection can be set to a value
    {
        GRN g;
        double weight_value = 3.14;
        g.set_all_H2O(weight_value);
        for(const auto& node : g.get_H2O())
            for(const auto& weight : node)
                assert(weight - weight_value < 0.00001 &&
                       weight - weight_value > -0.00001);
    }

    //All weights in H2H connection can be set to a value
    {
        GRN g;
        double weight_value = 3.14;
        g.set_all_H2H(weight_value);
        for(const auto& node : g.get_H2H())
            for(const auto& weight : node)
                assert(weight - weight_value < 0.00001 &&
                       weight - weight_value > -0.00001);
    }

    //All output nodes can be set to true or false
    {
        GRN g;
        bool nodes_state = false;
        g.set_all_out_nodes(nodes_state);
        for(auto node : g.get_output_nodes())
            assert(node == nodes_state);
    }
    //All output nodes treshold can be set to a value
    {
        GRN g;
        double treshold_value = 3.14;
        g.set_all_out_tresh(treshold_value);
        for (const auto& treshold : g.get_out_tresh())
        {
            assert(treshold - treshold_value < 0.00001 &&
                   treshold - treshold_value > -0.00001);
        }
    }
    //All hidden nodes can be set to true or false
    {
        GRN g;
        bool nodes_state = false;
        g.set_all_hid_nodes(nodes_state);
        for(auto node : g.get_hidden_nodes())
            assert(node == nodes_state);
    }

    //Output nodes can be updated based on values of hidden nodes
    {
        GRN g;
        bool hidden_nodes_value = true;
        //all hidden nodes are 1
        g.set_all_hid_nodes(hidden_nodes_value);
        //all weights H20 are
        //hidden_nodes_value / number_of_connections_per_output_node
        double weight_value = hidden_nodes_value;
        weight_value /= g.get_H2O().size();

        g.set_all_H2O(weight_value);
        //Therefore each output node will receive a signal of hidden_nodes_value
        g.set_all_out_tresh(hidden_nodes_value - 0.00001);
        hid_updates_out(g);
        for (size_t i = 0; i != g.get_output_nodes().size(); i++)
            assert(g.get_output_nodes()[i]);
    }

    //Given the initial inputs the hidden nodes will recieve a signal
    {
        GRN g;
        double inputs_value = 3.14; //in this case all inputs are the same
        auto inputs = std::vector<double>(g.get_input_nodes().size(),inputs_value);
        g.set_inputs(inputs);
        g.set_all_hid_tresh(inputs_value - 0.00001);
        //all weight I2H are
        //1 / number_of_connections_of each_hidden_to_input
        g.set_all_I2H( 1.0 / g.get_I2H().size() );
        auto inp_signals = inp_updates_hid(g);
        //Therefore each hidden node will receive a signal of hidden_nodes_value
        for (size_t i = 0; i != inp_signals.size(); i++)
            assert(inp_signals[i] - inputs_value < 0.00001 &&
                   inp_signals[i] - inputs_value > -0.00001);

    }

    //  //The values of the hidden nodes are stored in a vector to be used in
    //  //the successive timestep to auto-update the hidden nodes
    ////At initialization the values in this vector are all 0
    //  {
    //    GRN g;
    //    for (const auto& value : g.get_past_hid_val())
    //      {
    //        assert(value < 0.000001 && value > -0.000001)
    //      }
    //    assert(store_hid_values(g) != g.set_all_hid_nodes(1));
    //  }
    //The hidden nodes updated themselves toghether with input
    {
        GRN g;
        bool hidden_value = true; //in this case all nodes are the same
        g.set_all_hid_nodes(hidden_value);
        //all weight H2H are
        //1 / number_of_connection_to_each_hidden_node
        g.set_all_H2H( 1.0 / g.get_H2H().size());
        auto hid_signals = hid_updates_hid(g);
        //Therefore each hidden node will receive a signal == hidden_value
        for (size_t i = 0; i != hid_signals.size(); i++)
            assert(hid_signals[i] - static_cast<double>(hidden_value) < 0.00001 &&
                   hid_signals[i] - static_cast<double>(hidden_value) > -0.00001);
        //if instead the weights are smaller signal != hidden_value
        g.set_all_H2H(1.0 / g.get_H2H().size() * 2);
        hid_signals = hid_updates_hid(g);
        for (size_t i = 0; i != hid_signals.size(); i++)
            assert( hid_signals[i] - static_cast<double>(hidden_value) > 0.00001 ||
                    hid_signals[i] - static_cast<double>(hidden_value) < -0.00001);

    }

    //Given three initial input a GRN gives back an output
    {
        GRN g;//the weights and connection and nodes are all to 0, so output should be 0
        std::vector<double> inputs{100,100,100};
        g.set_inputs(inputs);
        jordi_response_mech(g);
        for(auto output : g.get_output_nodes())
        {
            assert(output);
        }
    }

    //The number of connections of a network can be counted
    {
        GRN g;
        assert(n_connections(g) == static_cast<int> (
                   g.get_H2H().size() * g.get_H2H()[0].size() +
               g.get_H2O().size() * g.get_H2O()[0].size() +
                g.get_I2H().size() * g.get_I2H()[0].size()
                )
                );

    }

    //The weights of a network can be summed toghether
    {
        GRN g;
        g.set_all_H2O(1);
        g.set_all_I2H(1);
        g.set_all_H2H(1);
        auto sum = weights_sum(g);
        auto n_con = static_cast<double>(n_connections(g));
        assert(sum - n_con < 0.00001 &&
               sum - n_con > -0.00001);
    }

    //The mean of the weights of the network can be calculated
    {
        GRN g;
        double weights_value = 1;
        g.set_all_H2O(weights_value);
        g.set_all_I2H(weights_value);
        g.set_all_H2H(weights_value);
        assert(weights_mean(g) - weights_value < 0.00001 &&
               weights_mean(g) - weights_value > -0.00001);
    }

    //The variance of weights of a network can be calculated
    {
        GRN g;
        double weights_value = 1;
        //All weights are the same so variance should be 0
        g.set_all_H2O(weights_value);
        g.set_all_I2H(weights_value);
        g.set_all_H2H(weights_value);
        assert(weights_var(g) < 0.00001 &&
               weights_var(g) > -0.00001);
    }

    //A GRN can be printed and loaded from an ifstream and ofstream file
    {
        GRN g{1,2,1};
        g.set_all_H2O(0.12);
        g.set_all_I2H(0.123);
        g.set_all_H2H(0.124);
        const std::string filename = "grn.csv";
        save_grn(g, filename);
        GRN g1;
        g1 = load_grn(filename);
        assert( g == g1);
    }
#endif

    //A GRN can be initialized by a vector of inds
    //specifying how many nodes will compose each layer
    {
        std::vector<int> layers{1,2,3};
        GRN g{layers};
        for(int i  = 0; i != g.get_n_layers(); i++)
        {
            assert(g.get_layers()[i].get_nodes().size() == static_cast<unsigned int>(layers[i]));
        }

    }
}
