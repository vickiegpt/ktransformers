#!/bin/bash
set -e  

# clear build dirs
rm -rf build
rm -rf *.egg-info
rm -rf csrc/build
rm -rf csrc/ktransformers_ext/build
rm -rf csrc/ktransformers_ext/cuda/build
rm -rf csrc/ktransformers_ext/cuda/dist
rm -rf csrc/ktransformers_ext/cuda/*.egg-info
rm -rf ~/.ktransformers
echo "Installing python dependencies from requirements.txt"
python -m pip install -r requirements-local_chat.txt
python -m pip install -r ktransformers/server/requirements.txt
echo "Installing ktransformers"
KTRANSFORMERS_FORCE_BUILD=TRUE python -m pip install -v . --no-build-isolation
python -m pip install third_party/custom_flashinfer/

# SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
# echo "Copying thirdparty libs to $SITE_PACKAGES"
# cp -a csrc/balance_serve/build/third_party/prometheus-cpp/lib/libprometheus-cpp-*.so* $SITE_PACKAGES/
# patchelf --set-rpath '$ORIGIN' $SITE_PACKAGES/sched_ext.cpython*


echo "Installation completed successfully"