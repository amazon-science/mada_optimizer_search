## MADA: Meta-Adaptive Optimizers through hyper-gradient Descent

**Authors: Kaan Ozkara, Can Karakus, Parameswaran Raman, Mingyi Hong, Shoham Sabach, Branislav Kveton, Volkan Cevher**

This repository includes the code to simulate experiments for our paper [MADA: Meta Adaptive Momentum Estimates through Hypergradient Descent] (https://arxiv.org/abs/2401.08893). The GPT training code is based on nanoGPT by Andrej Karpathy (https://github.com/karpathy/nanoGPT). Meta optimizer implementation is inspired by (https://github.com/kach/gradient-descent-the-ultimate-optimizer/tree/main).

`./config` includes configuration files that controls the parameters in the code.

`./results` includes some of the results that were mentioned in the quip document for the project.

`./gdtuo.py` is the implementation of meta optimizer through hypergradient descent.

`./model.py` includes a generic GPT-2 type implementation from nanoGPT.

`./plot... .py` files are used to plot the results that are in ./results.

`train.py`, `train_ddp.py`, `toy.py`, `toy2.py`,  includes the files to run experiments. 

`train_ddp.py` is the latest run file and has from scratch supoorts for ddp, gradient_accumulation.

Example run:

`python train_ddp.py config/train_gpt2_small.py --dtype='float32' --beta1=0.9 --beta2=0.95 --beta3=0.0 --rho=0.6 --c=1.0 --gamma=1.0`

The arguments here refer to the initial values of the optimizer parameters. Additional variables about the nanoGPT run can also be included if needed for e.g. to determine  logging, grad accumulation and so on. At the moment, to change the hypergradient hyperparameters (such as learning rate) and ddp size one, please update the code. The output directory to save log files is set as a FSx directory and would need to be changed inside the code as well. There are two types of logging, the first one where for every `log_iter` the optimizer parameters, training loss and validation loss are logged. The second one is logging at the end of run.

## Citation

Please consider citing our paper if you use our code:
```text
@misc{ozkara2024mada,
      title={MADA: Meta-Adaptive Optimizers through hyper-gradient Descent}, 
      author={Kaan Ozkara and Can Karakus and Parameswaran Raman and Mingyi Hong and Shoham Sabach and Branislav Kveton and Volkan Cevher},
      year={2024},
      eprint={2401.08893},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

