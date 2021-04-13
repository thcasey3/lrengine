===============
Version History
===============

Version 0.0.5
-------------
* Added map_to_frame() method
* bug fixes, updated docs, updated unittests

Version 0.0.4
-------------
* Added on_date() and in_range() methods for narrowing the dates to specific dates or ranges
* Added "only_unique" option to reduce_dates() for keeping only unique dates when lists
* Added "null" output for function errors in engine
* Renamed "names" column of frame to "name" for consistency with other columns
* updated examples, documentation, unittests, and fixed some bugs

Version 0.0.3
-------------
* Expanded date parsing to look for all possible combinations of number sequences
* Expanded find_dates() and reduce_dates() accordingly to deal with lists
* Expanded the list of date formats that can be found
* Improved pattern searching with dict
* Fixed bug in engine
* Fixed missing 's' parameter in sea()
* Improved exception handling


Version 0.0.2
-------------
* Added option to use a dict and re expressions to define patterns
* Added new methods **find_patterns()** and **reduce_names()**
* some code cleanup
* added to unittests
* improved documentation and template script

Version 0.0.1
-------------
* First release



