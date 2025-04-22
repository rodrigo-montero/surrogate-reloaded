# Surrogate Reloaded
### Framework for the testing and validation of surrogate models.

---
## Parking Agent

### Training data
All the training data from the DRL agent can be found in the folder:
```
logs/her/parking-v0_1/json_log
```

### Creating the dataset and classifier
This data is used to create the heldout dataset:
```
python -m indago.avf.train --algo her --exp-id 1 --avf-policy mlp --env-name park --env-id parking-v0 --build-heldout-test --test-split 0.2 --seed 0
```

Which in turn is used for training the classifier:
```
python -m indago.avf.train --algo her --env-name park --env-id parking-v0 --exp-id 1 --test-split 0.3 --avf-policy mlp --training-progress-filter 50 --oversample 0.0 --n-epochs 2000 --learning-rate 3e-4 --batch-size 256 --patience 10 --weight-loss --layers 1 --seed 988478156 --heldout-test-file heldout-set-seed-0-0.2-split-5-filter-cls.npz"
```

### Running the MLP
To run the original MLP model (for reference) we can use the launch.json (Test - Original GA Parking). We can also run it from the command line:
```
python -m indago.experiments --algo her --exp-id 1 --env-name park --env-id parking-v0 --avf-train-policy mlp --avf-test-policy ga_saliency_rnd --failure-prob-dist --num-episodes 10 --num-runs-each-env-config 1 --training-progress-filter 50 --layers 4 --budget 3 --num-runs-experiments 1
```

The above command will run the original classifier with the GA algorithm to create a bunch of new agent environments.

### Creating a new surrogate

The following steps will show how to add a new classifier to the system. This is to get you integrated with the code so you understand how the surrogate model classes tick.

First we create a new file for the model:
```
indago/avf
```
In the above folder you'll find the files for the surrogates.

Here we have avf_cnn.py and avf_mlp.py. The cnn file has been copied and modified from the avf_mlp.py file.

This new 'cnn' class must then be added to the factories.py file. In this file, you'll notice that it has been added in the get_avf_policy method.

Next we take a look at the avf_policy.py file. This is the file that defines the model layers. Notice the mlp model has 4 distinct variances. These are called with the --layer argument in the above python snippets.

Finally we have the dataset.py file, which contains the dataset specific imformation. In this file we added a flag for the 'cnn' policy type, on lines 104 and 115.

NOTE: The dataset file is setup to use tabular data and so the models at the moment need to have the correct number of input feature. This is the file where you can make some major changes to the dataset if you wish to have a different set of input features for your model.

The input should always be some form of environment data, with the target being the pass or fail of the environment.

The model must always output a classification value, where the value is between 0.0 and 1.0.

### Testing the new model

To test the model, you can run the same commands as above, but instead of --avf-train-policy mlp, you use --avf-train-policy cnn.

