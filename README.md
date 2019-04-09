TRNpy: Parallelized TRNSYS simulation with Python
=================================================
Simulate TRNSYS deck files in serial or parallel and use parametric tables to
perform simulations for different sets of parameters. TRNpy helps to automate
these and similar operations by providing functions to manipulate deck files
and run TRNSYS simulations from a programmatic level.


Usage
=====
TRNpy can be used as a standalone application or imported into other Python
scripts. Please note that a its core it simply makes the command line call
`TRNExe.exe your_deck_file.dck`. You still need your own, properly licensed,
copy of the software TRNSYS to make this work. TRNpy is merely a wrapper
or API to include TRNSYS simulations into your Python workflow.

Standalone TRNpy
----------------
TRNpy can be compiled into a Windows .exe file with the script `setup_exe.py`
and the following tips are valid for `trnpy_script.py` and `trnpy_script.exe`.

* By double-clicking the program, the main() function of this script is
  executed. It performs the most common tasks possible with this application.

    * The first file dialog allows to choose one or multiple deck files to
      simulate.
    * The second file dialog allows to choose a parametric table, which can be
      an Excel or a csv file. The first row must contain names of parameters
      you want to change. The following rows must contain the values of those
      parameters for each simulation you want to perform. TRNpy will make
      the substitutions in the given deck file and perform all the simulations.
    * You can cancel the second dialog to perform regular simulations.
    * The parametric table could look like this, to modify two parameters
      defined in TRNSYS equations:

        | Parameter_1 | Parameter_2 |
        | ----------- | ----------- |
        | 100         | 0           |
        | 100         | 1           |
        | 200         | 0           |
        | 200         | 1           |

* Running the program from a command line gives you more options, because
  you can use the built-in argument parser.

    * Type `python trnpy_script.py --help` or `trnpy_script.exe --help` to see
      the help message and an explanation of the available arguments.
    * This allows e.g. to enable parallel computing, hide the TRNSYS windows,
      suppress the parametric table file dialog, define the folder where
      parallel simulations are performed, and some more.
    * Example command:

        `trnpy_script.exe --parallel --copy_files --n_cores 4 --table disabled`

* Creating a shortcut to the executable is another practical approach

    * Arguments can be appended to the path in the `Target` field
    * Changing the field `Start in` to e.g. `C:\Trnsys17\Work` will always
      open the file dialogs in that folder

Module Import
-------------
Import `trnpy.core` into your own Python script. There you can
initialize objects of the `DCK_processor()` and `TRNExe()` classes and use
their functions. The first can create `dck` objects from regular TRNSYS
input (deck) files and manipulate them, the latter can run simulations with
the given `dck` objects.
This also gives you the option to perform post-processing tasks with
the simulation results (something that cannot be automated in the standalone
version).


Installation
============

Windows Executable
------------------
If you received the precompiled `trnpy_script.exe`, just save the complete
application folder anywhere. As explained, it makes sense to
create shortcuts to the executable from your TRNSYS work folders.
It can be compiled with `setup_exe.py`.

Python
------
If you want to use Python but do not yet have it installed, the easiest way to
do that is by downloading and installing **Anaconda** from here:
https://www.anaconda.com/download/
It's a package manager that distributes Python with data science packages.

During installation, please allow to add variables to `$PATH` (or do that
manually afterwards.) This allows Python to be started via command line from
every directory, which is very useful.

With an existing Python environment, you can use `setup.py` to install
TRNpy into the appropriate place in your Python environment. Then you can
import it into your own Python scripts.


Support
=======
For questions and help, contact Joris Nettelstroth.
If you would like to request or contribute changes, do the same.
