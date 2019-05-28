chinese, english, digit OCR using dnn
#### [install warp-ctc]
git clone --recursive https://github.com/jpuigcerver/pytorch-baidu-ctc.git
cd pytorch-baidu-ctc

after line 72, add a new line
extra_compile_args["cxx"].append("-g0")

python setup.py build
python setup.py install

cd $PYTHON_PATH/site-packages
cp -r torch_baidu_ctc-0.2.1-py3.7-linux-x86_64.egg/torch_baidu_ctc ./