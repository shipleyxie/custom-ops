# it will show many LOG message
export TF_CPP_MIN_VLOG_LEVEL=3
export TF_CPP_MIN_LOG_LEVEL=0

# clean work space and compile pip package
make clean
make zero_out_pip_pkg || exit 1

# the previous step will generate .whl in artifacts
# you need to ACTIVATE conda env devAudio before install
# and MUST uninstall pre-installed tensorflow-custom-ops
# or you will not use the new compiled one.
pip uninstall tensorflow-custom-ops -y
pip install artifacts/tensorflow_custom_ops-0.0.1-cp37-cp37m-linux_x86_64.whl
