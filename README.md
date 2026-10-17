Para que o sistema funcione corretamente foi necessário realizar a linkagem da biblioteca blas no diretório de include/ do spack.
Isso foi feito seguindo o comando:

```
cd ~/spack/var/spack/environments/trab02/.spack-env/view/include
ln -s $(spack location -i openblas)/include/openblas/cblas.h .
ln -s $(spack location -i openblas)/include/openblas/openblas_config.h .
```