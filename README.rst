.. raw:: html

   <h1 align="center">

AIBenchmark

.. raw:: html

   </h1>

.. raw:: html

   <h2 align="center">

Benchmark your model against other models

.. raw:: html

   </h2>

About
-----

AIBenchmark is a package which lets you quickly get the benchmark of
your model based on the popular datasets and compare with existing
leaderboard. It also has a nice collection of metrics which you could
easily import.

We currently support 14 text-based and 2 image-based datasets for
AutoBenchmarking aiming for regression/classification tasks. Available
datasets could be found in aibenchmark/dataset.py file.

Or run the following code:

.. code:: python


   from aibenchmark.dataset import DatasetsList

   print(list(DatasetsList.get_available_datasets()))

Code example for benchmarking:

.. code:: python

   from aibenchmark.benchmark import Benchmark
   from aibenchmark.dataset import DatasetInfo, DatasetsList


   benchmark = Benchmark(DatasetsList.Texts.SST)
   dataset_info: DatasetInfo = benchmark.dataset_info
   print(dataset_info)

   test_features = dataset_info.data['Texts']
   model = torch.load(...)
   # Implement your code based on the type of model you use, your pre- and post-processing etc.
   outputs = model.predict(test_features)

   # Results of your model based on predictions
   benchmark_results = benchmark.run(predictions=outputs, metrics=['accuracy', 'precision', 'recall', 'f1_score']) 

   # Metrics
   print(benchmark_results)
   # Existing leaderboard for this dataset
   print(benchmark.get_existing_benchmarks())

Features
--------

1) Fast comparison of metrics of your model and other SOTA models for
   particular dataset
2) Supporting 16+ most populat datasets, the list is always updating.
   Soon we willl support more than 1000 datasets
3) All metrics in one place and we are adding new ones in a standardised
   way

Starting
--------

.. code:: bash

   # Clone this project
   $ pip install git+https://github.com/BasedLabs/aibenchmark

Technologies
------------

The following tools were used in this project:

-  `Pytorch <https://pytorch.org/>`__
-  `Transformers <https://huggingface.co/transformers>`__
-  `Scikit-learn <https://scikit-learn.org/stable/>`__

