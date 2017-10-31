# trl

[![Build Status](https://travis-ci.org/tyrion/trl.svg?branch=master)](https://travis-ci.org/tyrion/trl)

## Prerequisites

* This project requires Python 3 to run.
* Additionally ensure you have build_essential and python3-dev installed in order for Theano to run properly.

## Install

Clone the repository:
```bash
$ git clone https://github.com/tyrion/trl.git
$ cd trl
```

Make a virtualenv to install dependencies (Optional):
```bash
$ mkvirtualenv -p /usr/bin/python3 -a `pwd` trl
```

Install the requirements:
```bash
$ pip install -r requirements.txt
```
Install the project:
```bash
$ pip install -e .
```

## Run the tests

Install the test dependencies:
```bash
$ pip install -e .[test]
```

Run the tests:
```bash
$ pytest
```
