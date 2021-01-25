# TIMe

Code for DASFAA2021-Topological Interpretable Multi-Scale Sequential Recommendation

We implement TIMe  based on 

[NeuRec]: https://github.com/wubinzzu/NeuRec	"NeuRec"

Firstly, download this code and unpack the downloaded source to a suitable location.

install tensorflow2.0 library：

- no GPU

```bash
conda install tensorflow
```

- GPU
```bash
conda install tensorflow-gpu
```

Secondly, go to '*./TIMe*' and compline the evaluator of cpp implementation with the following command line:
```bash
conda install Cython
```

```bash
python setup.py build_ext --inplace
```

If the compilation is successful, the evaluator of cpp implementation will be called automatically.
Otherwise, the evaluator of python implementation will be called.

**Note that the cpp implementation is much faster than python.**

install dependencies：

```bash
conda install pandas
conda install scikit-learn
conda install networkx
conda install matplotlib
conda install treelib
conda install tqdm
```
Thirdly, specify dataset and recommender in configuration file *default.properties*.

Finally, run [main.py](./main.py) in IDE or with command line:

```bash
python main.py --recommender=TIMe
```
