# ExpFamily

Unix | Windows | CodeCov | License
---- | ------- | ------- | -------
[![Travis](https://travis-ci.org/tlienart/ExpFamily.jl.svg?branch=master)](https://travis-ci.org/tlienart/ExpFamily.jl) | [![AppVeyor](https://ci.appveyor.com/api/projects/status/github/tlienart/ExpFamily.jl?branch=master&svg=true)](https://ci.appveyor.com/project/tlienart/expfamily-jl) | [![CodeCov](http://codecov.io/github/tlienart/ExpFamily.jl/coverage.svg?branch=master)](http://codecov.io/github/tlienart/ExpFamily.jl?branch=master) | [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

ExpFamily.jl is a package designed to provide efficient manipulation of objects belonging in an exponential family (e.g.: Gaussians). This is expected to be particularly useful for methods such as Nonparametric BP or Expected Propagation where one needs to manipulate such distributions at every step of the algorithm.

Requirements:

* Julia >= 0.5
