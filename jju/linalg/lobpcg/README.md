# Locally Optimal Block Preconditioned Conjugate Gradient

Algorithms in this module are from [Duersch _et. al_](https://epubs.siam.org/doi/abs/10.1137/17M1129830).

```bibtex
@article{duersch2018robust,
  title={A robust and efficient implementation of LOBPCG},
  author={Duersch, Jed A and Shao, Meiyue and Yang, Chao and Gu, Ming},
  journal={SIAM Journal on Scientific Computing},
  volume={40},
  number={5},
  pages={C655--C676},
  year={2018},
  publisher={SIAM}
}
```

## Notes

- `A_norm` and `B_norm` from original are calculated based on `A(x0)`, but the formula used is for Gaussian distributed inputs.
- `polynomial` evaluation for gradients seems highly inefficient.

## TODO

- Ortho implementation
- jvp?
- fixed number of iterations implementation?
- Basic implementation with non-None `B` and `iK`
