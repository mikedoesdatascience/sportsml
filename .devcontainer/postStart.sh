#!/bin/bash

if ! grep -q "alias ltr='ls -lhtr'" ~/.bashrc; then
    echo "alias ltr='ls -lhtr'" >> ~/.bashrc
fi