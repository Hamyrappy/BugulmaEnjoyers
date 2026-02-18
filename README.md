# BugulmaEnjoyers

We code respect into every token of the Tatar language

## Installation

### Clone the repository

```bash
git clone https://github.com/Hamyrappy/BugulmaEnjoyers.git 
```

### Install pixi ([https://pixi.sh/dev/installation/](https://pixi.sh/dev/installation/))

Linux/MacOS:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

### Then install dependencies

```bash
pixi install
```

## Run

### Approaches used in the article

- mT0 (Tatar prompt)
	```bash
	pixi run python one_step.py -vv -f test_inputs.tsv -o outputs_mT0_tatar.tsv --batch-size 10
	```

- mT0 (English prompt)
	```bash
	pixi run python one_step.py -vv -f test_inputs.tsv -o outputs_mT0_eng.tsv --batch-size 10 -l en
	```

- Gemini Pro (Tatar)
	```bash
	pixi run python one_step.py -vv -f test_inputs.tsv -o outputs_gemini.tsv --batch-size 40
	```

- mT0 + Gemini Pro
	```bash
	pixi run python main.py -vv -f test_inputs.tsv -o outputs_mT0_gemini.tsv --batch-size-1 10 --batch-size-2 40
	```

- mT0 + vocab deletion
	```bash
	pixi run python main.py -vv -f test_inputs.tsv -o outputs_mT0_vocab.tsv --batch-size-1 10 --batch-size-2 40 --detoxifier-2 vocab
	```

Results will be saved in the specified output file.
