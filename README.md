# Constrained adversarial attacks

Experimental setup for introducing constraints to universal adversarial attacks.

### Usage

Install dependencies

```
pip install -r requirements.txt
```

Run all experiments for all configuration options:

```
make all
```

Help running the experiments

```
python3 -m exp --help
```

### Development instructions

First ensure all dev dependencies are installed

```
pip install -r requirements-dev.txt
```

Available code quality checks

```
make test    -- Run unit tests
make lint    -- Run linter
make dev     -- Test and lint, all at once
```