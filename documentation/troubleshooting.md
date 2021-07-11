## Troubleshooting on Error: ImportError: /lib64/libstdc++.so.6: 

If you get this error: ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by <environmentpath>/lib/python3.8/site-packages/faiss/_swigfaiss.so)

run:
```
LD_LIBRARY_PATH=<environmentpath>/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
```
*(This happened on one of our servers, no idea why) thanks to: https://stackoverflow.com/questions/20357033/usr-lib-x86-64-linux-gnu-libstdc-so-6-version-cxxabi-1-3-8-not-found*

## unable to open shared memory object

When training with tas-balanced, sometimes this error pops up - but it does not stop training, so it can be disregarded:

Traceback (most recent call last):
  File ".../python3.8/multiprocessing/queues.py", line 239, in _feed
    obj = _ForkingPickler.dumps(obj)
  File ".../python3.8/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File ".../python3.8/site-packages/torch/multiprocessing/reductions.py", line 319, in reduce_storage
    metadata = storage._share_filename_()
RuntimeError: unable to open shared memory object </torch_16261_3802747475> in read-write mode