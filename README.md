# BugulmaEnjoyers

Кодируем уважение в каждый токен татарского языка

## Insatllation

### Склонируйте репозиторий

```bash
git clone https://github.com/Hamyrappy/BugulmaEnjoyers.git 
```

### Поставьте pixi ([https://pixi.sh/dev/installation/](https://pixi.sh/dev/installation/))

Linux/MacOS:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

### Затем установите зависимости

```bash
pixi install
```

## Run

```python
pixi run python main.py -vv -f test_inputs.tsv -o test_outputs.tsv
```

Результаты будут в `test_outputs.tsv`
