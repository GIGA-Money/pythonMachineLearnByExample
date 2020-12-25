# this folder contains a tutorial from toward data science articles.
* pysyft_tut: https://towardsdatascience.com/federated-learning-3097547f8ca3
    * install: needs pytorch 1.4 (as of jul 2020) use: 
        * conda install pytorch=1.4 torchvision cudatoolkit=10.2 -c pytorch
            * cuda really is a must, however there are additional steps involved with having the gpu work:
                * see: https://www.youtube.com/watch?v=6gk7giKER6s along with text based linked in vid.
        * pip install syft --user
        * works for win install (no conda comparability yet)
        * pysyft (currently) uses an old version of h5py that is only compatible with py 3.6
