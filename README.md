
This is the official implementation of the paper "[Integrating GNN and Neural ODEs for Estimating Non-Reciprocal Two-Body Interactions in Mixed-Species Collective Motion](https://openreview.net/forum?id=qwl3EiDi9r)", a conference paper in NeurIPS 2024.

To see the presentation of the paper, please visit [here](https://neurips.cc/virtual/2024/poster/93465).

## Requirements

Before installing this library, make sure you have the deep graph library (DGL) installed. You can find the installation instructions [here](https://www.dgl.ai/pages/start.html).

To install the library, run:

```pip install -e . ```

at this folder. This will install the required dependencies.

Afterward, you can add the library to your Python path by running:

```python setup.py develop```


## The Harmonic Interaction Model


### Training

To run the simulation for the harmonic interaction model and train the obtained data, run the codes in the IPython notebook `harmonic_interaction_model.ipynb`.

This will generate the data and the model trained on the data in the directory with the name you specify.

### Evaluation

To visualize the data, run the codes in the IPython notebook `harmonic_interaction_model_visualize.ipynb`.
By specifying the directory where the data is saved, you can visualize the movement of the particles in the data.

To quantify the performance of the trained model, run the codes in the IPython notebook `harmonic_interaction_model_calcError.ipynb`.
By specifying the directory where the model is saved, you can calculate the error of the model, i.e. Mean Squared Error (MSE) and Mean Absolute Error (MAE).

To plot the predicted functions in the trained model, run the codes in the IPython notebook `harmonic_interaction_model_plots.ipynb`.
By specifying the directory where the data and the model are saved, you can plot the functions in the trained model against the true functions.

To run a simulation with the trained model, run the codes in the IPython notebook `harmonic_interaction_model_reSimulate.ipynb`.
By specifying the directory where the model is saved, you can run a simulation with the trained model.
The results of this simulation can be visualized by running the codes in the IPython notebook `harmonic_interaction_model_visualize.ipynb`.

## The Mixed-Species Model

### Training

To run the simulation for the mixed-species model and train the obtained data, run the codes in the IPython notebook `mixed_species_model.ipynb`.

This will generate the data and the model trained on the data in the directory with the name you specify.

### Evaluation

To visualize the data, run the codes in the IPython notebook `mixed_species_model_visualize.ipynb`.
By specifying the directory where the data is saved, you can visualize the movement of the particles in the data.

To quantify the performance of the trained model, run the codes in the IPython notebook `mixed_species_model_calcError.ipynb`.
By specifying the directory where the model is saved, you can calculate the error of the model, i.e., Mean Squared Error (MSE) and Mean Absolute Error (MAE).

To plot the predicted functions in the trained model, run the codes in the IPython notebook `mixed_species_model_plots.ipynb`.
By specifying the directory where the data and the model are saved, you can plot the functions in the trained model against the true functions.

To run a simulation with the trained model, run the codes in the IPython notebook `mixed_species_model_reSimulate.ipynb`.
By specifying the directory where the model is saved, you can run a simulation with the trained model.
The results of this simulation can be visualized by running the codes in the IPython notebook `mixed_species_model_visualize.ipynb`.

## Results

Our model achieves the performance shown in the figures and the supplemental tables in the paper.
