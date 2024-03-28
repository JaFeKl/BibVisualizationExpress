[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



# BibVisualizationExpress


BibVisualizationExpress is a small Python library for quick visualizing bibliographic data.
The library uses .csv data which can either be provided by file
or directly crawled from scopus by using [pybliometrics](https://github.com/pybliometrics-dev/pybliometrics), a Scopus API for python.  

Currently this library provides three functions for visualization:

`plot_records_over_time()` 


<img src="images/example_records_over_time.png" alt="RecordsOverTime" width="400" height="300">

`plot_world_map_country_count()`

<img src="images/example_world_map_country_count.png" alt="WorldMapCountryCount" width="500" height="200">

`plot_keyword_worldcloud()`

<img src="images/example_keyword_worldcloud.png" alt="KeywordWorldcloud" width="300" height="300">

## Install
Easiest to install all dependencies is using conda:
``` 
conda env create -f environment.yml  
conda activate bibvis
```

## Usage
Check out the [example notebook](examples/generate_figures.ipynb), which gives a good overview.

## Contributing
If you like the library feel free to contribute!

## License
MIT license, see [LICENSE](LICENSE)
