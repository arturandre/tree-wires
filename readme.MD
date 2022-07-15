# Detecting tree and wire entanglements with deep learning

This repository contains code and data for the experiments from the paper *Detecting tree and wire entanglements with deep learning*.

The paper can be seen [here.](https://link.springer.com/article/10.1007/s00468-022-02305-0#Sec2)

This work proposes a deep learning approach to detect trees entangled to power and communication lines using street-level imagery. Testing the method was performed using 1001 images from urban trees from the cities of São Paulo and Porto Alegre (both in Brazil). We found an overall accuracy of 74.6% (73.6% for São Paulo and 75.6% for Porto Alegre), suggesting that the methodology could be suitable in the future for city management to avoid risks of accidents due to contact between trees and electrical wiring.

## Using the code

The current version of the repository allows:

To use a pre-trained network or to train a new network based on a pre-trained network to predict labels for the test dataset.

For convenience we make available the pre-trained model (mobilenetv2_test94) and the trained one based on in (mobilenetv2_test98).

We also make available the training dataset.

To train a new network use the command:
`python classifier.py t mobilenetv2_test94 --savecheckpoint mobilenetv2_test98 --dataaug --finetune '11' --val_split '0.2' --customclassifier 1080,540 --numepochs 100`

To predict labels for the test dataset:

- with the pre-trained network use the command:

`python classifier.py p mobilenetv2_test94 --test --customclassifier 1080,540`

- with the trained network use the command:

`python classifier.py p mobilenetv2_test98 --test --customclassifier 1080,540`

## Acknowledgements

This research is part of the INCT of the Future Internet for Smart Cities funded by CNPq proc. 465446/2014-0, Coordenação de Aperfeiçoamento de Pessoal de Nível Superior-Brasil (CAPES) -Finance Code 001, FAPESP proc. 14/50937-1, and FAPESP proc. 15/24485-9. This study was financed in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior–Brasil (CAPES) –Finance Code 001 and in part by the grant #2018/10767-0, São Paulo Research Foundation (FAPESP).

## Reference

If you find this useful for your research, please consider citing the following work.

```
@article{oliveira2022detecting,
  title={Detecting tree and wire entanglements with deep learning},
  author={Oliveira, Artur Andr{\'e} and Buckeridge, Marcos S and Hirata, Roberto},
  journal={Trees},
  pages={1--13},
  year={2022},
  publisher={Springer}
}
```

Feel free to contact us with any comments or feedback.