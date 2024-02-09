# DTCC DeepFacade

Get pixel coordinates of windows in a facade from street view image

Highlights:

* Uses `pyproject.toml` and installable using `pip install`
* Uses `scikit_build_core` as build backend
* C++ extensions built via `CMake` and `CMakeLists.txt`

This project is part the
[Digital Twin Platform (DTCC Platform)](https://gitlab.com/dtcc-platform)
developed at the
[Digital Twin Cities Centre](https://dtcc.chalmers.se/)
supported by Sweden’s Innovation Agency Vinnova under Grant No. 2019-421 00041.

## Documentation

* [Introduction](./docs/introduction.md)
* [Installation](./docs/installation.md)
* [Usage](./docs/usage.md)
* [Development](./docs/development.md)

## Authors (in order of appearance)

* [Anders Logg](http://anders.logg.org)
* [Vasilis Naserentin](https://www.chalmers.se/en/Staff/Pages/vasnas.aspx)

## License

DTCC DeepFacade is licensed under the
[MIT license](https://opensource.org/licenses/MIT).

Copyright is held by the individual authors as listed at the top of
each source file.

## Community guidelines

Comments, contributions, and questions are welcome. Please engage with
us through Issues, Pull Requests, and Discussions on our GitHub page.



### Temporary docs

## Install: 

- linux:
    - python3 -m venv deepfacade && source deepfacade/bin/activate
    - pip install  -r requirements.txt

- mac:
    - python3 -m venv deepfacade && source deepfacade/bin/activate
    - pip install  -r requirements.txt

- windows:
    - python3 -m venv deepfacade
    - deepfacade\Scripts\activate
    - pip install  -r requirements.txt


## Poetry 
- install poetry: `curl -sSL https://install.python-poetry.org | python3 -`
- create venv: `poetry shell`
- activate venv: `source $(poetry env info --path)/bin/activate`
- install libs: `poetry install`
- Build and install as a library:
    ```
    poetry build
    pip install dist/{wheel_file}.whl
    ```
- add poetry to path: `export  PATH="~/.local/bin:$PATH"` to ~/.bashrc
- Error "`Failed to unlock the collection!`": 
    - run -> `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`

- find ~/.cache/pypoetry -name '*.lock' -type f -delete


## ultralytics
- yolo predict model=yolov8x-oiv7.pt source="cmp_b0001.jpg" conf=0.1
- yolo predict model=yolov8l-oiv7.pt source="cmp_b0001.jpg" conf=0.1

## Todo:
- hf: choose the window size base on the size of the image and make it consistent across different image sizes. - fixit!!!
- detection on a directory of images - done
- return window coordinates or save them to file (json?) - done
- option to plot windows in another directory - done
- update main file
- test and run as a library.

