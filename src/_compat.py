"""
Python 3.14 compatibility patches. Import this module before using `datasets`.

In Python 3.14, pickle._Pickler._batch_setitems gained a required `obj`
parameter that datasets 3.x's custom Pickler subclass doesn't accept.
"""
import sys

if sys.version_info >= (3, 14):
    import datasets.utils._dill as _hf_dill
    import dill as _dill_lib

    def _compat_batch_setitems(self, items, obj=None):
        if self._legacy_no_dict_keys_sorting:
            return super(_hf_dill.Pickler, self)._batch_setitems(items, obj)
        try:
            items = sorted(items)
        except Exception:
            from datasets.fingerprint import Hasher
            items = sorted(items, key=lambda x: Hasher.hash(x[0]))
        _dill_lib.Pickler._batch_setitems(self, items, obj)

    _hf_dill.Pickler._batch_setitems = _compat_batch_setitems
