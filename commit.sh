#!/bin/sh

git add *.py *.pyx *.pxd requirements.txt *.sh
git commit
git push -u origin main