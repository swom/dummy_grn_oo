#ifndef GRN_H
#define GRN_H
#include "layer.h"
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>

class GRN
{
public:
    GRN(size_t n_input = 3,
        size_t n_hidden = 3,
        size_t n_output = 2,
        double weights = 0.1);

    GRN(std::vector<std::vector<double>>ConI2H,
        std::vector<std::vector<double>>ConH2H,
        std::vector<std::vector<double>>ConH2O,
        std::vector<double>THidden,
        std::vector<double>TOutput,
        std::vector<bool>ExHidden,
        int n_input,
        int n_output)
    ;

    GRN(std::vector<int> nodes_per_layer);

    ///Gets the constant reference to connection from input to hidden layer
    const std::vector<std::vector<double> >& get_I2H() const noexcept {return m_ConI2H;}

    ///Gets the reference to connection from input to hidden layer
    std::vector<std::vector<double> >& get_I2H() noexcept {return m_ConI2H;}

    ///Gets the const reference to connection from hidden to hidden layer
    const std::vector<std::vector<double> >& get_H2H() const noexcept {return m_ConH2H;}

    ///Gets the reference to connection from hidden to hidden layer
    std::vector<std::vector<double> >& get_H2H() noexcept {return m_ConH2H;}

    ///Gets the const ref to connection from hidden to output layer
    const std::vector<std::vector<double> >& get_H2O() const noexcept {return m_ConH2O;}

    ///Gets the reference connection from hidden to output layer
    std::vector<std::vector<double> >& get_H2O() noexcept {return m_ConH2O;}

    ///Gets const reference to input layer states
    const std::vector<double>& get_input_nodes() const noexcept {return  m_ExInput;}

    ///Gets const reference to input layer states
    std::vector<double>& get_input_nodes() noexcept {return  m_ExInput;}

    ///Gets const reference hidden layer states
    const std::vector<bool>& get_hidden_nodes() const noexcept {return  m_ExHidden;}

    ///Gets reference hidden layer states
    std::vector<bool>& get_hidden_nodes() noexcept {return  m_ExHidden;}

    ///Gets const ref to tresholds of hidden nodes
    const std::vector<double>& get_hid_tresh() const noexcept {return m_THidden;}

    ///Gets ref to tresholds of hidden nodes
    std::vector<double>& get_hid_tresh() noexcept {return m_THidden;}

    ///Gets const ref to the vector of layers
    const std::vector<layer>& get_layers() const noexcept {return m_layers;}

    ///Gets the size of the layer vector
    int get_n_layers() const noexcept{ return static_cast<int>(m_layers.size());}

    ///Gets const reference output layer states
    const std::vector<bool>& get_output_nodes() const noexcept {return  m_ExOutput;}

    ///Gets reference output layer states
    std::vector<bool>& get_output_nodes() noexcept {return  m_ExOutput;}

    ///Gets const reference to sporulation output (m_ExOutput[0])
    bool get_output_spo() const noexcept {return m_ExOutput[0];}

    ///Gets const ref to tresholds of output nodes
    const std::vector<double>& get_out_tresh() const noexcept {return m_TOutput;}

    ///Gets ref to tresholds of output nodes
    std::vector<double>& get_out_tresh() noexcept {return m_TOutput;}

    ///Sets all weights of H2O to a given value
    void set_all_H2H(double value) noexcept;

    ///Sets all weights of H2O to a given value
    void set_all_H2O(double value) noexcept;

    ///Sets all weights of I2H to a given value
    void set_all_I2H(double value) noexcept;

    ///Sets an output node to true or false
    void set_hid_node(int index_node, bool state);

    ///Sets an output node to true or false
    void set_out_node(int index_node, bool state);

    ///Sets all output nodes to a state: T/F
    void set_all_hid_nodes(bool state);

    ///Sets all output nodes to a state: T/F
    void set_all_out_nodes(bool state);

    ///Sets all the tresholds of the output nodes to a certain value
    void set_all_out_tresh(double value);

    ///Sets all the tresholds of the hidden nodes to a certain value
    void set_all_hid_tresh(double value);

    ///Sets inputs to the values given by a vector
    ///(that is created in individual in working circumstances)
    void set_inputs(std::vector<double> inputs);

private:
    std::vector<std::vector<double> > m_ConI2H;	// Connections from Input to Hidden layer
    std::vector<std::vector<double> > m_ConH2H;	// Connections from Hidden to Hidden layer
    std::vector<std::vector<double> > m_ConH2O;	// Connections from Hidden to Output layer

    std::vector<double> m_THidden;			// Threshold hidden nodes
    std::vector<double> m_TOutput;			// Threshold output nodes

    std::vector<double> m_ExInput;		// Expression during 'development'
    std::vector<bool> m_ExHidden;			// Expression during 'development'
    std::vector<bool> m_ExOutput;			// Expression during 'development'

    /// NEW ARCHITECTURE
    /// This part will eventually substitute the previous private members
    /// and override all functions associated with them

    std::vector<layer> m_layers;
};

///Compares two GRNs to see if all their states,
/// except input and output nodes are the same
bool operator==(const GRN& lhs, const GRN& rhs);

///Checks two GRNs to see if at least one of their states,
/// except input and output nodes are not the same
bool operator!=(const GRN& lhs, const GRN& rhs);

///Loads a grn from a text file
/// ATTENTION EXTREMELY FRAGILE
/// For now it only works if the stream is at the right point in the file
/// and the network saved in the file is the same as the network in the
/// program
/// NO CHECKS are implemented to make sure this is the case
/// it will need a parser to do so
std::ifstream& operator>>(std::ifstream& is, GRN& p);

///Writes weights and tresholds and states to a stream
///  in the following order:
/// I2H,H2H,H2O,THidden,TOutput
/// It does not save nor inputs or outputs states
/// As they are not relevant and could just cause problems
std::ostream& operator<<(std::ostream& os, const GRN& p);

///Gets the layer weights from a ifstream file
std::vector<double> load_node_weights(std::ifstream& is);

///Input nodes update the states of output nodes
std::vector<double> hid_updates_hid(GRN& g) noexcept;

///Hidden nodes update the states of output nodes
void hid_updates_out(GRN& g) noexcept;

///Input nodes update the states of output nodes
std::vector<double> inp_updates_hid(GRN& g) noexcept;

///The GRN reads inputs and gives back the outputs
///!!!!****Implemented as in Jordi's model***!!! I do not like it
void jordi_response_mech(GRN& g);


///Terrible functions for loading weights from a streamfile
std::vector<std::vector<double>> load_I2H_weights(std::ifstream& is);
std::vector<std::vector<double>> load_H2H_weights(std::ifstream& is);
std::vector<std::vector<double>> load_H2O_weights(std::ifstream& is);
std::vector<double> load_hid_tresh(std::ifstream& is);
std::vector<double> load_out_tresh(std::ifstream& is);
std::vector<bool> load_hid_state(std::ifstream& is);
int load_n_input_nodes(std::ifstream& is);
int load_n_output_nodes(std::ifstream& is);


///Mutates weights of connections between input and hidden layer
///For now requires to get distribution and rng from somewhere else
/// (simulation)
void mutation_I2H(GRN& g, std::minstd_rand& rng,
                  std::bernoulli_distribution& mu_p,
                  std::normal_distribution<double> mu_st) noexcept;

///Mutates weights of connections between input and hidden layer
///For now requires to get distribution and rng from somewhere else
/// (simulation)
void mutation_H2H(GRN& g, std::minstd_rand& rng,
                  std::bernoulli_distribution& mu_p,
                  std::normal_distribution<double> mu_st) noexcept;

///Mutates weights of connections between input and hidden layer
///For now requires to get distribution and rng from somewhere else
/// (simulation)
void mutation_H2O(GRN& g, std::minstd_rand& rng,
                  std::bernoulli_distribution& mu_p,
                  std::normal_distribution<double> mu_st) noexcept;

///Mutates weights of thresholds of hidden layer
///For now requires to get distribution and rng from somewhere else
/// (simulation)
void mutation_hid_tr(GRN& g, std::minstd_rand& rng,
                     std::bernoulli_distribution& mu_p,
                     std::normal_distribution<double> mu_st) noexcept;

///Mutates weights of thresholds of output layer
///For now requires to get distribution and rng from somewhere else
/// (simulation)
void mutation_hid_tr(GRN& g, std::minstd_rand& rng,
                     std::bernoulli_distribution& mu_p,
                     std::normal_distribution<double> mu_st) noexcept;


///Applies random mutation to weights and thresholds of an individual
void mutation(GRN& g, std::minstd_rand& rng,
              std::bernoulli_distribution& mu_p,
              std::normal_distribution<double> mu_st) noexcept;

///Counts the number of connections in a network
int n_connections(const GRN& g) noexcept;

///Functions to print to a stream file the weight, tresholds and states
/// of the network
std::ostream& save_I2H_weights(std::ostream& os, const GRN& p);
std::ostream& save_H2H_weights(std::ostream& os, const GRN& p);
std::ostream& save_H2O_weights(std::ostream& os, const GRN& p);
std::ostream& save_THidden(std::ostream& os, const GRN& p);
std::ostream& save_TOutput(std::ostream& os, const GRN& p);
std::ostream& save_ExHidden(std::ostream& os, const GRN& p);
std::ostream& save_n_input_nodes(std::ostream& os, const GRN& grn);
std::ostream& save_n_output_nodes(std::ostream& os, const GRN& grn);
void save_grn(const GRN& grn, const std::string& filename);

///Sums the weights of hidden to hidden connections
double sum_I2H(const GRN& g) noexcept;

///Sums the weights of hidden to hidden connections
double sum_H2H(const GRN& g) noexcept;

///Sums the weights of hidden to ouput connections
double sum_H2O(const GRN& g) noexcept;

///Updates hidden nodes states based on inut and hidden nodes themselves
void update_hid(GRN& g) noexcept;

///Calculates the sum of all weights
double weights_sum (const GRN& g) noexcept;

///Calculates the mean value of all weights
double weights_mean (const GRN& g) noexcept;

///Calculates the variance of weights in a GRN
double weights_var(const GRN& g) noexcept;



void test_GRN();
#endif // GRN_H
