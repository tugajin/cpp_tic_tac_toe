g++ -O3 -Wall -shared -std=c++17 -fPIC -DNDEBUG `python3 -m pybind11 --includes` gamelibs.cpp -o gamelibs`python3-config --extension-suffix`
