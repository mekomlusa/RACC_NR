# RACC_NR Project (with U-Net support)

Forked from [RACC_NR](https://github.com/pulakeshupadhyaya/RACC_NR) with add-in support for U-Net.

# U-Net code

To enable training in U-Net, simply refer to the `DNN_uNet_auto.py` script. Type `python DNN_uNet_auto.py --help` for help. The code is tested under Python 3.5 and should be applicable for version 3.5 and above.

Sample usage:

* To train given a training and a validation dataset: model persistence will be saved to the path specified under the `-d` flag.

```
python DNN_uNet_auto.py train -trgt ../unet/4095_randomized/training/html/ -trn ../unet/4095_noisy_randomized_0p01/training/html/ -vgt ../unet/4095_randomized/validation/html/ -vn ../unet/4095_noisy_randomized_0p01/validation/html/ --ft html -d ../unet/results/
```

* To predict given a test dataset and a saved model persistence: every single prediction will be saved to the path specified under the `-d` flag.

```
python DNN_uNet_auto.py test -tgt ../unet/4095_randomized/testing/html/ -tn ../unet/4095_noisy_randomized_0p01/testing/html/ --ft html -d ../unet/predictions/train_0p01/test_0p01/html/ -p ../unet/results/html_unet_nopooling.h5
```

Note: in order to train, one **MUST** specify both training and testing dataset. As for testing, the flag `-p` is required to specify where the saved model persistence is. Below is the list of accept parameters:

```
RACC_NR U-Net script

positional arguments:
  <command>             'train' or 'test'

optional arguments:
  -h, --help            show this help message and exit
  -trgt /path/to/dataset/, --traingt /path/to/dataset/
                        Ground truth folder for the training data
  -trn /path/to/dataset/, --trainn /path/to/dataset/
                        Noisy folder for the training data
  -vgt /path/to/dataset/, --valgt /path/to/dataset/
                        Ground truth folder for the validation data
  -vn /path/to/dataset/, --valn /path/to/dataset/
                        Noisy folder for the validation data
  -tgt /path/to/dataset/, --testgt /path/to/dataset/
                        Ground truth folder for the testing data
  -tn /path/to/dataset/, --testn /path/to/dataset/
                        Noisy folder for the testing data
  --ft html             Available file type: html, latex, jpeg, pdf
  -d /path/to/prediction/, --dest /path/to/prediction/
                        Path to save the prediction files (at the prediction
                        time) or model file (at the training time)
  -p /path/to/persistence/, --persistence /path/to/persistence/
                        Path to the saved model file
```


# training code
4095_filetypes_cnn.py is the program for training a file type recognition system of 4095 bit segments of files.

DNN_conv_deconv.py is used to train a convolution-deconvolution network to provide soft information using
natural redundancy(NR) which can be supplemented with LPDC decoding for error correction.

# testing code
DNN_end_to_end_test.py tests an end to end system which does file type recognition based on the system trained by
4095_filetypes_cnn.py,and based on the result, uses the corresponding convolution-deconvolution network trained by
DNN_conv_deconv.py. It then stores the soft information so that it is available for the new LDPC decoder

LDPC_ED_BSC.cpp tests the new LDPC decoding aided by NR-based soft information stored by DNN_end_to_end_test.py
