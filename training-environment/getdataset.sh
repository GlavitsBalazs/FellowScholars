#!/bin/sh

wget -c http://datashare.is.ed.ac.uk/download/DS_10283_2791.zip
mkdir DS_10283_2791
cd DS_10283_2791
unzip ../DS_10283_2791.zip
rm ../DS_10283_2791.zip
find . -name '*.zip' -exec unzip {} \;
rm *.zip