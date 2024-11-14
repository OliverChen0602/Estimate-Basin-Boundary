# Estimate-Basin-Boundary

This is a documentation of all code(.ipynb) files and all data files(.csv)

## Brief summary for all files

1. arbitrary_2d_new.ipynb: The code file for synthetic 2D system.
2. triple_well_potential.ipynb: The code file for triple well potential.
3. three_populations.ipynb: The code file for three population competition.
4. dataset_arbi2d.general_10000.csv: Data file for 10000 points sampled uniformly at random from domain. The x0
and y0 columns are coordinates, while attracted = +1 or -1 indicates whether this point belongs to the basin
of attraction of the studied attractor.
5. dataset_arbi2d_near_5000_x.csv: Data file for 5000 points that are close to actual basin
boundary, generated by bisection method. The threshold to stop bisection is specified by x.
6. dataset_triplewell.general_10000.csv: Data file for 10000 points sampled uniformly at random from domain. The x0
and y0 columns are coordinates, while attracted = +1 or -1 indicates whether this point belongs to the basin
of attraction of the studied attractor.
7. dataset_triplewell_near_5000_x.csv: Data file for 5000 points that are close to actual basin
boundary, generated by bisection method. The threshold to stop bisection is specified by x.
8. dataset_threepop.general_15000.csv: Data file for 15000 points sampled uniformly at random from domain. The x0, y0
and z0 columns are coordinates, while attracted = +1 or -1 indicates whether this point belongs to the basin
of attraction of the studied attractor.
9. dataset_triplewell_near_3200_x.csv: Data file for 3200 points that are close to actual basin
boundary, generated by bisection method. The threshold to stop bisection is specified by x.
10. deprecated: files that are no longer used.

Remark: All datasets are collected independently of each other.

## Supplementary information for code implementation

Note that the code files all share the same set of implementation, except that for three population competition model,
the input dimension of the network is adjusted to 3. Therefore, for each relevant code block for arbitrary_2d_new.ipynb,
there are substantial comments to elaborate the operations. Nevertheless, we list important functions here:

1. *system* \
arguments: (x,y) - 2D coordinates of a point \
output: xp, yp - relevant time derivatives

2. *bisection* \
arguments: a, b - two points; delta - a threshold \
output: (a, b) - two points with different label and with distance smaller than delta

3. *euclidean_distance* \
arguments: point1, point2 - two points \
output: L2-norm between them

4. *is_attracted* \
arguments: x,y - 2D coordiantes of a point \
output: a Boolean to indicate whether a point eventually is attracted by the attractor.

5. *simulation* \
arguments: x0, y0 - 2D coordinates of a point \
output: a Boolean to determine whether the trajectory, starting from the point argument as the initial point,
converges to the attractor.

6. *train_model_bce* \
arguments: net - neural network to train; dataset_train - labelled training set; dataset_validation1 - validation set
coming from dataset_xxxx_general_xxxx.csv; dataset_validation00x - validation set coming from dataset_xxxx_near_0.0x.csv; dataset_validation010 - validation set coming from dataset_xxxx_near_0.1.csv; batchsize - size of batch of data to
be loaded by Dataloader; epochs - number of epochs to train; lr - learning rate \
output: no output, but invoke plot_contour every 20% of epoch, track training and validation accuracy and loss.

Remark: Training set and validation sets of desired size are uniformly drawn from the relevant data files. Note that there is no overlapping between ANY two different sets of data.

7. *plot_contour* \
arguments: net - trained neural network \
output: a plot of decision boundary

8. *test_model* \
arguments: net - trained neural network; dataset_test: test set \
output: accuracy

9. *train_model_hinge* \
arguments: same as 6 \
output: same as 6

10. *train_model_hinge_square* \
arguments: same as 6 \
output: same as 6

11. *dynamical_system_vector* \
same function as 1, but just deals with tensor input and output.

12. *custom_loss_function* \
arguments: model - network; batch_inputs - a batch of points(sampled close to decision boundary); var - variance of
Gaussian PDF to approximate Dirac delta function \
outputs: mean of dynamic-informed loss term across the batch.

13. *find_decision_boundary_points_bisection* \
arguments: f - network; target_count - number of points to be sampled; threshold - upper bound of distance to the actual
DECISION boundary of the network \
outputs: a batch of points close to decision boundary

14. *train_model_dynamical_adaptive* \
arguments: same as 6 \
outputs: same as 6
