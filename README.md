# Constrained adversarial attacks

Experimental setup for introducing constraints to universal adversarial attacks.

### Usage

Install dependencies

```
pip install -r requirements.txt
```

Run all experiments, for all configuration options:

```
make exp
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
make all     -- Test and lint, all at once
```