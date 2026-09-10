# Vendored Python (optional)

`electron-builder` copies this directory into the packaged app as
`Contents/Resources/python`, and the app prefers `python/bin/siftr` over
anything on `PATH`.

It is empty by default. Bundling the Python side means shipping torch
(~590 MB), onnxruntime (~80 MB) and their dependencies — a ~1.5 GB app before
any model is downloaded. For a personal build it is usually better to
`pip install 'siftr[ui,faces]'` once and let the app find it on `PATH`.

To vendor it anyway, install a standalone (not virtualenv) Python here along
with siftr and its extras, so that `python/bin/siftr` runs without depending on
an interpreter elsewhere on the machine. A plain `.venv` copied here will *not*
work: its `pyvenv.cfg` points at the base interpreter it was created from.
