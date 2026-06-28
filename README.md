# ExecuTorch Engine for DJL

As of the most recent version (0.36.0), DJL only supports the TorchScript export API.  This API has been deprecated for several point release versions
of PyTorch.  One of the export APIs present in PyTorch is the [ExecuTorch](https://executorch.ai/) backend.  This is a lightweight integration layer
that was built to be cross-language and is intended for edge model deployment.

The goal of this project is to provide an ExecuTorch engine for DJL such that PyTorch based models can make use of the newer export APIs.
As a separate engine, it also allows for slow migration from TorchScript/PyTorch to this new backend.