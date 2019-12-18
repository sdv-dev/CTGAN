Command Line Interface
======================

**CTGAN** comes with a command line interface that allows modeling and sampling data
without the need to write any Python code. This is available under the ``ctgan`` command
that will have been set up in your system upon installing **CTGAN**.

Working with CSV files
----------------------

In the simplest scenario, ``ctgan`` can be used to read data from a CSV file, with headers,
and using ``,`` as the separator.

For this, you can call ``ctgan`` passing it only two arguments:

* The path to the input data file.
* The path to the output file.

.. code-block:: bash

   $ ctgan path/to/input.csv path/to/output.csv

Optionally, if the CSV file contains no header, an additional ``--no-header`` option can be added.

.. code-block:: bash

   $ ctgan path/to/input.csv path/to/output.csv --no-header

The commands above will model the data from the input CSV file and produce a new CSV file with
synthetic data and as many rows as there were in the input table.


JSON Metadata
-------------

The previous example will work for datasets that only contain continuous columns, but if the
table that you are trying to model contains any discrete columns you will need to provide an
additional Metadata file which specifies which columns are continuous and which ones are discrete.

.. code-block:: bash

   $ ctgan path/to/input.csv path/to/output.csv -m path/to/metadata.json

This Metadata file will be in JSON format, containing an entry called ``columns``, with a list
of sub-documents containing both the ``name`` of the column and its ``type``::

   {
       "columns": [
           {
               "name": "age",
               "type": "continuous"
           },
           {
               "name": "workclass",
               "type": "categorical"
           },
           {
               "name": "fnlwgt",
               "type": "continuous"
           },
           ...
       ]
   }

.. note:: Column types can be ``continuous`` for continuous columns and ``categorical``,
          ``ordinal`` or ``discrete`` for non-continuous columns.

Alternatively, if there is no Metadata file, the list of discrete column names can be provided
as a comma separated list, without white spaces, directly on the command line:

.. code-block:: bash

   $ ctgan path/to/input.csv path/to/output.csv -d 'one_column,another_column,yet_another_column'

.. note:: If the input CSV contains no header and the ``--no-header`` option is used, the column
          names will be the integer indices. This applies to both the Metadata JSON and the
          command line ``-d`` option.


TSV: Tab Separated Values
-------------------------

**CTGAN** CLI also supports files in TSV format which are text files where each row represents a
row in the table and columns are separated by tabs or white spaces, without header names::

    100        A        True
    200        B        False
    105        A        True
    120        C        False
    ...        ...        ...

In this case, the metadata filename has to also be stored as a TSV file, which describes each
column as one line.

Each line starts with either a ``C`` or ``D`` character, which represent continuous or discrete
columns respectively.

* For continuous columns, the following two numbers indicate the range of the column.
* For discrete columns, the following strings indicate all the possible values in the column.

For example, the metadata file for the table shown above would be::

    C    0    500
    D    A    B    C
    D    True     False


In order to use this format, the ``--tsv`` option must be added to the command, and the output
will also be stored in TSV format:

.. code-block:: bash

   $ ctgan path/to/input.dat path/to/output.dat -m path/to/input.meta --tsv


Additional Options
------------------

There are a couple of additional options that can be passed to the ``ctgan`` command to
control how the data is modeled and sampled:

* ``-e, --epochs [EPOCHS]``: Number of training epochs to perform. Defaults to 300.
* ``-n, --num-samples [NUM_SAMPLES]``: Number of samples to generate. Defaults to the training
  data size.

.. code-block:: bash

   $ ctgan path/to/input.csv path/to/output.csv -e 100 -n 1000
