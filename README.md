# Working through https://course.fast.ai/, feb 2025, with Recurse study group

[Paul Winkler](https://www.recurse.com/directory/5804-paul-winkler)


## Notes on Chapter 2: Production

### Installing fastbook / fastai on a Macbook Air M1 (2020)

Various warnings about Mac not being supported.

As a starting point, I had:

- Python 3.12 installed via homebrew
- Python 3.13 likewise
- I chose 3.12 for this as there seems to be a general recommendation to use one less
  than latest python for Pytorch et al?

I use [direnv](https://direnv.net/) to manage all my directory-specific environment setup.
You don't have to - you could manually use pyvenv, virtualenv, or whatever you like;
with direnv, I just create this `.envrc` file in the directory where I clone
this repo:

```
layout python python3.12
```

After creating that file and doing the usual `direnv allow` command, a
virtualenv is created for me with the right version of python:

```console

$ vim .envrc
direnv: error /Users/paul/src/practical-deep-learning-fastai/.envrc is blocked. Run `direnv allow` to approve its content

$ direnv allow
direnv: loading ~/src/practical-deep-learning-fastai/.envrc
direnv: export +VIRTUAL_ENV ~PATH

(python-3.12)$ which python
/Users/paul/src/practical-deep-learning-fastai/.direnv/python-3.12/bin/python
```

Now I'm ready to install packages.

First I followed [this blog](https://pnote.eu/notes/pytorch-mac-setup/#mps--metal-support)
which suggested the default install steps for pytorch:

```console
$ pip3 install torch torchvision torchaudio
```

That seemed to work as per the example program:

```pycon
>>> import torch
>>> # Check that MPS is available
>>> if not torch.backends.mps.is_available():
...    if not torch.backends.mps.is_built():
...        print("MPS not available because the current PyTorch install was not "
...              "built with MPS enabled.")
...    else:
...        print("MPS not available because the current MacOS version is not 12.3+ "
...              "and/or you do not have an MPS-enabled device on this machine.")
... else:
...    print("all good")
all good
```

So I tried `pip install fastbook` (which includes fastai) - apparently this reinstalls
different versions of pytorch et al.
Why am I using pip?
Because that's what's used in all the example jupyter notebooks for this course
on Kaggle and Colab, and it works there. I'm comfortable with Pip and didn't feel a need
to go on a tangent into trying different Python installers and getting up to
speed on mamba vs conda et al.
Good news - Pip seems to have worked!

```console

$ pip install fastbook
Collecting fastbook
  Downloading fastbook-0.0.29-py3-none-any.whl.metadata (13 kB)
... [LOTS OF OUTPUT SNIPPED]
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torchaudio 2.6.0 requires torch==2.6.0, but you have torch 2.5.1 which is incompatible.
Successfully installed aiohappyeyeballs-2.4.6 aiohttp-3.11.12 aiosignal-1.3.2 annotated-types-0.7.0 anyio-4.8.0 appnope-0.1.4 argon2-cffi-23.1.0 argon2-cffi-bindings-21.2.0 arrow-1.3.0 asttokens-3.0.0 async-lru-2.0.4 attrs-25.1.0 babel-2.17.0 beautifulsoup4-4.13.3 bleach-6.2.0 blis-1.2.0 catalogue-2.0.10 certifi-2025.1.31 cffi-1.17.1 charset-normalizer-3.4.1 click-8.1.8 cloudpathlib-0.20.0 comm-0.2.2 confection-0.1.5 contourpy-1.3.1 cycler-0.12.1 cymem-2.0.11 datasets-3.3.1 debugpy-1.8.12 decorator-5.1.1 defusedxml-0.7.1 dill-0.3.8 executing-2.2.0 fastai-2.7.18 fastbook-0.0.29 fastcore-1.7.29 fastdownload-0.0.7 fastjsonschema-2.21.1 fastprogress-1.0.3 fonttools-4.56.0 fqdn-1.5.1 frozenlist-1.5.0 fsspec-2024.12.0 graphviz-0.20.3 h11-0.14.0 httpcore-1.0.7 httpx-0.28.1 huggingface-hub-0.28.1 idna-3.10 ipykernel-6.29.5 ipython-8.32.0 ipython-genutils-0.2.0 ipywidgets-7.8.5 isoduration-20.11.0 jedi-0.19.2 joblib-1.4.2 json5-0.10.0 jsonpointer-3.0.0 jsonschema-4.23.0 jsonschema-specifications-2024.10.1 jupyter-client-8.6.3 jupyter-core-5.7.2 jupyter-events-0.12.0 jupyter-lsp-2.2.5 jupyter-server-2.15.0 jupyter-server-terminals-0.5.3 jupyterlab-4.3.5 jupyterlab-pygments-0.3.0 jupyterlab-server-2.27.3 jupyterlab-widgets-1.1.11 kiwisolver-1.4.8 langcodes-3.5.0 language-data-1.3.0 marisa-trie-1.2.1 markdown-it-py-3.0.0 matplotlib-3.10.0 matplotlib-inline-0.1.7 mdurl-0.1.2 mistune-3.1.1 multidict-6.1.0 multiprocess-0.70.16 murmurhash-1.0.12 nbclient-0.10.2 nbconvert-7.16.6 nbformat-5.10.4 nest-asyncio-1.6.0 notebook-7.3.2 notebook-shim-0.2.4 overrides-7.7.0 packaging-24.2 pandas-2.2.3 pandocfilters-1.5.1 parso-0.8.4 pexpect-4.9.0 platformdirs-4.3.6 preshed-3.0.9 prometheus-client-0.21.1 prompt_toolkit-3.0.50 propcache-0.2.1 psutil-7.0.0 ptyprocess-0.7.0 pure-eval-0.2.3 pyarrow-19.0.0 pycparser-2.22 pydantic-2.10.6 pydantic-core-2.27.2 pygments-2.19.1 pyparsing-3.2.1 python-dateutil-2.9.0.post0 python-json-logger-3.2.1 pytz-2025.1 pyyaml-6.0.2 pyzmq-26.2.1 referencing-0.36.2 regex-2024.11.6 requests-2.32.3 rfc3339-validator-0.1.4 rfc3986-validator-0.1.1 rich-13.9.4 rpds-py-0.22.3 safetensors-0.5.2 scikit-learn-1.6.1 scipy-1.15.2 send2trash-1.8.3 sentencepiece-0.2.0 shellingham-1.5.4 six-1.17.0 smart-open-7.1.0 sniffio-1.3.1 soupsieve-2.6 spacy-3.8.4 spacy-legacy-3.0.12 spacy-loggers-1.0.5 srsly-2.5.1 stack_data-0.6.3 terminado-0.18.1 thinc-8.3.4 threadpoolctl-3.5.0 tinycss2-1.4.0 tokenizers-0.21.0 torch-2.5.1 torchvision-0.20.1 tornado-6.4.2 tqdm-4.67.1 traitlets-5.14.3 transformers-4.49.0 typer-0.15.1 types-python-dateutil-2.9.0.20241206 tzdata-2025.1 uri-template-1.3.0 urllib3-2.3.0 wasabi-1.1.3 wcwidth-0.2.13 weasel-0.4.1 webcolors-24.11.1 webencodings-0.5.1 websocket-client-1.8.0 widgetsnbextension-3.6.10 wrapt-1.17.2 xxhash-3.5.0 yarl-1.18.3
$

```

Apparently the warning about torchaudio version mismatch is OK enough? We'll see.


```pyshell

>>> import torch
>>> if not torch.backends.mps.is_available():
...     if not torch.backends.mps.is_built():
...         print("MPS not available because the current PyTorch install was not "
...               "built with MPS enabled.")
...     else:
...         print("MPS not available because the current MacOS version is not 12.3+ "
...               "and/or you do not have an MPS-enabled device on this machine.")
... else:
...     print("all ok")
...
all ok
>>> import fastbook
Matplotlib is building the font cache; this may take a moment.
>>>
```

### Running the notebook locally

I tried running `jupyter notebook` and navigating to chapter 2.
It worked! But cells wouldn't run. I noticed this logged in the terminal:

```console
[W 2025-02-18 16:39:30.338 ServerApp] Notebook book/01_intro.ipynb is not trusted
```

Jupyter will need to be told it's OK to trust this notebook, and then we can
run it:

```console
$ jupyter trust book/02_production.ipynb
Signing notebook: book/02_production.ipynb
```

Then we can run it again:

```console
$ juypter notebook
```

Once that's done, any notebook sections in the `book/` subdirectory
_that are intended to run locally_ should work.
I'll update my copies of the book chapter notebooks as I go through the course,
to ensure they work for me.

For example, I made some changes to my fork of the [chapter 2 notebook](book/02_production.ipynb)
to ensure I can resume locally with the
exported bear classification pickle (included in this repo) without having to
re-do the first half that's intended to be run on Colab with a GPU.  (I may
later try running _everything_ locally and see how long it takes to train on
the M1's GPU, which only has 8GB unified memory for both CPU and GPU!)

That done, it worked! Like so:

```python

from fastbook import *
from fastai.vision.widgets import *

path = Path()
exported_bear_model_path = path/'bear-export.pkl'
learn_inf = load_learner(exported_bear_model_path)

test_image_path = Path() / "images"
test_image_path = test_image_path / "grizzly.jpg"

learn_inf.predict(test_image_path')
```
```
('grizzly', tensor(1), tensor([2.7573e-04, 9.9949e-01, 2.3265e-04]))
```

### Enabling Voila

The book gives an obsolete command:

```
!jupyter serverextension enable --sys-prefix voila
```

That should be:

```
!jupyter server extension enable --sys-prefix voila
```

### Exporting with nbdev

This wasn't mentioned:

```console
$ pip install nbdev
````

The video code at 49:14 is wrong, this works:

```python
import nbdev.export
nbdev.export.nb_export("02_app.ipynb", name="deployable_is_it_a_cat")
```

That done, the exported file works when run locally:

```console
$ cd lessons/
$ python deployable_is_it_a_cat.py
* Running on local URL:  http://127.0.0.1:7861

To create a public link, set `share=True` in `launch()`.
```
I can click that URL and upload cats and dogs and get an answer:

<img src="./lessons/screenshot_is_it_a_cat.png">


### HUGGINGFACE WARNING: need to enable git lfs BEFORE adding a large blob


Don't just dump a big pickle file into your repo!
Other folks ran into this issue:

```
You will also need to install Git LFS, which will be used to handle large files
such as images and model weights.
```

We're going to need to install git-lfs first. On mac:

```
brew install git-lfs
```

Then we can enable it:

```
git lfs track '*.pkl'
git lfs track '*.jpg'
git lfs track '*.png'
git lfs track '*.jpeg'
```

Need to do that BEFORE adding big files.
