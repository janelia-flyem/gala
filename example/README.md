This directory contains simple examples to test the pipeline.  Admittedly,
this is a rather crude setup for verifying the correctness of the package.
To run the scripts or use the config files as is, RUN FROM THE GALA DIRECTORY.

There are three flows that can be tested:

gala-pixel
gala-train
gala-segmentation-pipeline

To run gala-pixel, one can use the config file already in the example
directory called 'pix-config.json'

In the ${GALA} directory you can call:

gala-pixel --config-file example/pix-config.json <some session name>

The config file points to grayscale data and a boundary classifier in the
example directory.  This command produces an ouptut log in the session
folder name and a prediction called STACK_prediction.h5 which is needed
for gala-train.  If you want to run the gala-train command as in the
example directory, copy this file to example/prediction.h5.  (This file
should have the same prediction as example/prediction_precomputed.h5.  One can copy
this precomputed prediction to prediction.h5 if one just wants to test
gala-train).


./example/train_command

This will call gala-train and put the results in train-sample.  It produces
an agglomeration classifier, agglom.classifier.h5 that should be copied
to example/agglom.classifier.h5.  (This file should have the same result
as in example/agglom.classifier_precomputed.h5.  One can copy this
precomputed classifier to agglom.classifier.h5 if one just wants to
test gala-segmentation-pipeline.)

gala-segmentation-pipeline --config-file example/seg-config.json <some session name>

This runs the segmentation pipeline and produces a labeled volume and log file.

Please examine the log files generated to ensure the output is similar to the output
stored in the example (i.e., check that the node numbers are the same).

This example will eventually be further simplified and converted into
a formal regression.
