.. _Spark:

Spark
=====

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2023

Tumult Core uses Spark as its underlying data processing
framework. This topic guide covers relevant information about Spark
for users of the Core library.

Configuring Spark sessions
--------------------------

Core uses :class:`Spark sessions <pyspark.sql.SparkSession>` to do all data processing operations.
As long as Spark has an active session, any calls that "create" new Spark
sessions use that active session.

If you want Core to use Spark sessions with any *specific* properties,
then before running Core code, you should create that Spark session:

.. code-block::

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.<your options here>.getOrCreate()

As long as this session is active, Core will use it.

.. dropdown:: Trouble Using Java 11 or higher?
    :container: + shadow
    :title: font-weight-bold text-danger text-center
    :body: bg-light
    :animate: fade-in-slide-down
    
    When using Java 11, Spark requires that Java be passed some additional configuration options.

    Not doing this before making calls to Tumult Core, or overwriting these configurations with your own, 
    will result in Spark raising ``java.lang.UnsupportedOperationException: sun.misc.Unsafe`` or ``java.nio.DirectByteBuffer.(long, int) not available`` 
    when evaluating queries. 
    
    Core attempts to set these options automatically, but if you are encountering issues, you may want to try:

    .. code-block::

        from tmlt.core.utils.configuration import get_java11_config

        spark = SparkSession.builder.config(conf=get_java11_config()).getOrCreate()

    instead of:

    .. code-block::

        spark = SparkSession.builder.getOrCreate()

    when initializing Spark. This is equivalent to:

    .. code-block::

        spark = (
            SparkSession.builder
            .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
            .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
            .getOrCreate()
        )


Connecting to Hive
^^^^^^^^^^^^^^^^^^

If you want to connect Spark to an existing `Hive <https://hive.apache.org/>`_
database, you should use the following options when creating a Spark session:

.. code-block::

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.<your options here>
        .config('spark.sql.warehouse.dir', '<Hive warehouse directory>')
        .enableHiveSupport()
        .getOrCreate()

To see where Hive's warehouse directory is, you can use the
`Hive CLI <https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Cli#LanguageManualCli-HiveInteractiveShellCommands>`_
(or its replacement,
`Beehive <https://cwiki.apache.org/confluence/display/Hive/HiveServer2+Clients#HiveServer2Clients-BeelineHiveCommands>`_)
to view the
`relevant configuration parameter <https://cwiki.apache.org/confluence/display/Hive/AdminManual+Metastore+3.0+Administration#AdminManualMetastore3.0Administration-GeneralConfiguration>`_:

.. code-block::

        > set hive.metastore.warehouse.dir;
        hive.metastore.warehouse.dir=/hive/warehouse

Materialization and data cleanup
--------------------------------

Tumult Core uses a Spark database (named "``tumult_temp_<time>_<uuid>``") to
materialize dataframes after noise has been added. This ensures that repeated
queries on a dataframe of results do not re-evaluate the query with fresh
randomness.

This has a few consequences for users:

* Queries are eagerly-evaluated, instead of lazily-evaluated.
* Operations create a temporary database in Spark.
* Core *does not* support multi-threaded operations, because the
  materialization step changes the active Spark database. (The active database is
  changed back to the original database at the end of the materialization step.)

Automatically cleaning up temporary data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tumult Core registers a cleanup function with ``atexit``
(see `Python's atexit documentation <https://docs.python.org/3/library/atexit.html>`_).
If a Spark session is still active when the program exits normally, this cleanup
function will automatically delete the materialization database.

If you wish to call ``spark.stop()`` before program exit, you should call
:func:`~tmlt.core.utils.cleanup.cleanup()` first. This will delete the materialization
database. This function requires an active Spark session, but is otherwise safe
to call at any time in a single-threaded program. (If
:func:`~tmlt.core.utils.cleanup.cleanup()` is called before a materialization step,
Core will create a new materialization database.)

Finding and removing leftover temporary data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The materialization database is stored as a folder in your Spark
warehouse directory.  If your program exits unexpectedly (for example,
because it was terminated with Ctrl-C),
or if the cleanup function is called without an active Spark session,
this temporary database (and its associated folder) may not be deleted.

Core has a function to delete any of these folders in the current
Spark warehouse: :func:`~tmlt.core.utils.cleanup.remove_all_temp_tables()`.
As long as your program is single-threaded, it is safe to call this function
at any time.

You can also manually delete this database by deleting its
directory from your Spark warehouse directory.
(If you did not explicitly configure a Spark warehouse directory,
look for a directory called ``spark-warehouse``.)
Spark represents databases as folders; the databases used
for materialization will be folders named "``tumult_temp_<time>_<uuid>``".
Deleting the folder will delete the database.

These folders are safe to manually delete any time that your program is not running.

Performance and profiling
-------------------------

All queries made with Core are executed by Spark. If you are having
performance problems, you will probably want to look at
`Spark performance-tuning options <https://spark.apache.org/docs/latest/sql-performance-tuning.html>`_.